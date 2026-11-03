import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

from cryptography.fernet import Fernet
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, String, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.auth.models import EmailOutbox
from dialogentis.auth.security import VerifiedUser, check_password, digest, rate_limit
from dialogentis.contracts import CursorPage, InputModel
from dialogentis.dependencies import Database
from dialogentis.models import Base, Membership, Organization, User

router = APIRouter(prefix="/api/v1", tags=["organizations"])
Role = Literal["admin", "editor", "viewer"]
Permission = Literal[
    "manage_members", "manage_admins", "transfer_ownership", "delete_organization", "manage_agents"
]
ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    "owner": frozenset(
        {
            "manage_members",
            "manage_admins",
            "transfer_ownership",
            "delete_organization",
            "manage_agents",
        }
    ),
    "admin": frozenset({"manage_members", "manage_agents"}),
    "editor": frozenset({"manage_agents"}),
    "viewer": frozenset(),
}


class Invitation(Base):
    __tablename__ = "invitations"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), index=True)
    email: Mapped[str] = mapped_column(String(320))
    role: Mapped[str] = mapped_column(String(16))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(default=False)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), index=True)
    actor_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(64))
    target_id: Mapped[UUID] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class OrganizationDeletion(Base):
    __tablename__ = "organization_deletions"
    job_id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(unique=True)
    requested_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
    )


class OrganizationInput(InputModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")


class RoleInput(InputModel):
    role: Role


class InvitationInput(RoleInput):
    email: EmailStr


class AcceptInput(InputModel):
    token: str = Field(min_length=20, max_length=512)


class TransferInput(InputModel):
    new_owner_user_id: UUID
    password: str = Field(min_length=12, max_length=128)


class OrganizationView(BaseModel):
    id: UUID
    name: str
    slug: str
    history_retention_days: int
    store_message_text: bool
    role: Literal["owner", "admin", "editor", "viewer"]


class AcceptedInvitationView(BaseModel):
    organization_id: UUID
    role: Literal["owner", "admin", "editor", "viewer"]


class MemberView(BaseModel):
    user_id: UUID
    email: str
    role: Literal["owner", "admin", "editor", "viewer"]


class MemberRoleView(BaseModel):
    user_id: UUID
    role: Role


class InvitationView(BaseModel):
    id: UUID
    email: str
    role: Role
    expires_at: datetime
    revoked: bool
    accepted: bool


def org_json(org: Organization) -> dict:
    return {
        "id": str(org.id),
        "name": org.name,
        "slug": org.slug,
        "history_retention_days": org.retention_days,
        "store_message_text": org.store_message_text,
    }


async def organization_context(
    db: Database,
    user: User,
    org_id: UUID,
    permission: Permission | None = None,
    *,
    lock: bool = False,
    allow_deleting: bool = False,
) -> Membership:
    membership = await db.get(Membership, (org_id, user.id))
    if membership is None:
        raise HTTPException(404, "Пространство не найдено")
    await db.execute(
        text("SELECT set_config('app.organization_id', :id, true)"), {"id": str(org_id)}
    )
    org_query = select(Organization).where(Organization.id == org_id)
    if lock:
        org_query = org_query.with_for_update()
    org = await db.scalar(org_query.execution_options(populate_existing=True))
    if org is None or (org.deletion_requested_at is not None and not allow_deleting):
        raise HTTPException(404, "Пространство не найдено")
    if lock:
        membership = await db.scalar(
            select(Membership)
            .where(
                Membership.organization_id == org_id,
                Membership.user_id == user.id,
            )
            .execution_options(populate_existing=True)
        )
        if membership is None:
            raise HTTPException(404, "Пространство не найдено")
    if permission and permission not in ROLE_PERMISSIONS.get(membership.role, frozenset()):
        raise HTTPException(403, "Недостаточно прав")
    return membership


def audit(db: Database, org_id: UUID, user: User, action: str, target: UUID):
    db.add(
        AuditEvent(
            organization_id=org_id,
            actor_id=user.id,
            action=action,
            target_id=target,
            created_at=datetime.now(UTC),
        )
    )


@router.get("/organizations", response_model=CursorPage[OrganizationView])
async def list_organizations(user: VerifiedUser, db: Database):
    rows = (
        await db.execute(
            select(Organization, Membership.role)
            .join(
                Membership,
                Membership.organization_id == Organization.id,
            )
            .where(Membership.user_id == user.id, Organization.deletion_requested_at.is_(None))
            .order_by(Organization.created_at, Organization.id)
        )
    ).all()
    return {"items": [{**org_json(org), "role": role} for org, role in rows], "next_cursor": None}


@router.post("/organizations", status_code=201, response_model=OrganizationView)
async def create_organization(body: OrganizationInput, user: VerifiedUser, db: Database):
    org_id = uuid4()
    await db.execute(
        text("SELECT set_config('app.organization_id', :id, true)"), {"id": str(org_id)}
    )
    org = Organization(id=org_id, name=body.name.strip(), slug=body.slug)
    if not org.name:
        raise HTTPException(422, "Укажите название пространства")
    try:
        async with db.begin_nested():
            db.add(org)
            await db.flush()
    except IntegrityError as exc:
        raise HTTPException(409, "Адрес пространства уже занят") from exc
    db.add(Membership(organization_id=org_id, user_id=user.id, role="owner"))
    audit(db, org_id, user, "organization.created", org_id)
    return {**org_json(org), "role": "owner"}


@router.get("/organizations/{org_id}", response_model=OrganizationView)
async def get_organization(org_id: UUID, user: VerifiedUser, db: Database):
    membership = await organization_context(db, user, org_id)
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(404, "Пространство не найдено")
    return {**org_json(org), "role": membership.role}


@router.get("/organizations/{org_id}/members", response_model=CursorPage[MemberView])
async def list_members(org_id: UUID, user: VerifiedUser, db: Database):
    await organization_context(db, user, org_id, "manage_members")
    rows = (
        await db.execute(
            select(Membership, User.email)
            .join(User)
            .where(
                Membership.organization_id == org_id,
            )
            .order_by(User.email)
        )
    ).all()
    return {
        "items": [{"user_id": str(m.user_id), "email": email, "role": m.role} for m, email in rows],
        "next_cursor": None,
    }


@router.patch("/organizations/{org_id}/members/{user_id}", response_model=MemberRoleView)
async def set_role(org_id: UUID, user_id: UUID, body: RoleInput, user: VerifiedUser, db: Database):
    actor = await organization_context(db, user, org_id, "manage_members", lock=True)
    target = await db.get(Membership, (org_id, user_id))
    if target is None:
        raise HTTPException(404, "Участник не найден")
    if target.role == "owner" or (
        "manage_admins" not in ROLE_PERMISSIONS[actor.role]
        and (target.role == "admin" or body.role == "admin")
    ):
        raise HTTPException(403, "Изменение этой роли недоступно")
    target.role = body.role
    audit(db, org_id, user, "member.role_changed", user_id)
    return {"user_id": str(user_id), "role": target.role}


@router.delete("/organizations/{org_id}/members/{user_id}", status_code=204)
async def remove_member(org_id: UUID, user_id: UUID, user: VerifiedUser, db: Database):
    actor = await organization_context(db, user, org_id, "manage_members", lock=True)
    target = await db.get(Membership, (org_id, user_id))
    if target is None:
        raise HTTPException(404, "Участник не найден")
    if target.role == "owner" or (
        "manage_admins" not in ROLE_PERMISSIONS[actor.role] and target.role == "admin"
    ):
        raise HTTPException(403, "Удаление этого участника недоступно")
    await db.delete(target)
    audit(db, org_id, user, "member.removed", user_id)


@router.post("/organizations/{org_id}/leave", status_code=204)
async def leave(org_id: UUID, user: VerifiedUser, db: Database):
    member = await organization_context(db, user, org_id, lock=True)
    if member.role == "owner":
        raise HTTPException(409, "Сначала передайте владение пространством")
    await db.delete(member)
    audit(db, org_id, user, "member.left", user.id)


@router.post("/organizations/{org_id}/transfer-ownership", status_code=204)
async def transfer(org_id: UUID, body: TransferInput, user: VerifiedUser, db: Database):
    owner = await organization_context(db, user, org_id, "transfer_ownership", lock=True)
    if not await check_password(user.password_hash, body.password):
        raise HTTPException(403, "Неверный пароль")
    target = await db.get(Membership, (org_id, body.new_owner_user_id))
    if target is None or target.user_id == user.id:
        raise HTTPException(422, "Выберите другого участника пространства")
    owner.role = "admin"
    await db.flush()
    target.role = "owner"
    audit(db, org_id, user, "organization.ownership_transferred", target.user_id)


def invitation_json(invitation: Invitation) -> dict:
    return {
        "id": str(invitation.id),
        "email": invitation.email,
        "role": invitation.role,
        "expires_at": invitation.expires_at.isoformat(),
        "revoked": invitation.revoked,
        "accepted": invitation.consumed_at is not None,
    }


async def send_invitation(
    db: Database, request: Request, invitation: Invitation, org: Organization
):
    key = request.app.state.settings.encryption_key
    if key is None:
        raise HTTPException(503, "Почтовый сервис не настроен")
    token = secrets.token_urlsafe(32)
    invitation.token_hash = digest(token)
    invitation.expires_at = datetime.now(UTC) + timedelta(days=7)
    link = f"{request.app.state.settings.public_base_url}/invitations/accept#token={token}"
    payload = json.dumps(
        {
            "to": invitation.email,
            "subject": "Приглашение в Dialogentis",
            "text": f"Вас приглашают в {org.name}.\n\n{link}",
        }
    )
    db.add(
        EmailOutbox(
            organization_id=org.id,
            encrypted_payload=Fernet(key.get_secret_value().encode())
            .encrypt(payload.encode())
            .decode(),
        )
    )


@router.post("/organizations/{org_id}/invitations", status_code=201, response_model=InvitationView)
async def invite(
    org_id: UUID, body: InvitationInput, request: Request, user: VerifiedUser, db: Database
):
    actor = await organization_context(db, user, org_id, "manage_members", lock=True)
    if "manage_admins" not in ROLE_PERMISSIONS[actor.role] and body.role == "admin":
        raise HTTPException(403, "Назначение администраторов доступно владельцу")
    email = str(body.email).casefold()
    await rate_limit(db, request, "invitation", f"{org_id}:{email}", 3, 60)
    member = await db.scalar(
        select(Membership)
        .join(User)
        .where(Membership.organization_id == org_id, User.email == email)
    )
    if member:
        raise HTTPException(409, "Пользователь уже состоит в пространстве")
    previous = (
        await db.scalars(
            select(Invitation).where(
                Invitation.organization_id == org_id,
                Invitation.email == email,
                Invitation.revoked.is_(False),
                Invitation.consumed_at.is_(None),
            )
        )
    ).all()
    if "manage_admins" not in ROLE_PERMISSIONS[actor.role] and any(
        item.role == "admin" for item in previous
    ):
        raise HTTPException(403, "Управление администраторами доступно владельцу")
    for item in previous:
        item.revoked = True
    invitation = Invitation(organization_id=org_id, email=email, role=body.role)
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(404, "Пространство не найдено")
    await send_invitation(db, request, invitation, org)
    db.add(invitation)
    await db.flush()
    audit(db, org_id, user, "invitation.created", invitation.id)
    return invitation_json(invitation)


@router.get("/organizations/{org_id}/invitations", response_model=CursorPage[InvitationView])
async def list_invitations(org_id: UUID, user: VerifiedUser, db: Database):
    await organization_context(db, user, org_id, "manage_members")
    rows = (
        await db.scalars(
            select(Invitation)
            .where(Invitation.organization_id == org_id)
            .order_by(Invitation.expires_at, Invitation.id)
        )
    ).all()
    return {"items": [invitation_json(i) for i in rows], "next_cursor": None}


@router.delete("/organizations/{org_id}/invitations/{invitation_id}", status_code=204)
async def revoke(org_id: UUID, invitation_id: UUID, user: VerifiedUser, db: Database):
    actor = await organization_context(db, user, org_id, "manage_members", lock=True)
    invitation = await db.get(Invitation, invitation_id)
    if invitation is None:
        raise HTTPException(404, "Приглашение не найдено")
    if "manage_admins" not in ROLE_PERMISSIONS[actor.role] and invitation.role == "admin":
        raise HTTPException(403, "Управление администраторами доступно владельцу")
    invitation.revoked = True
    audit(db, org_id, user, "invitation.revoked", invitation_id)


@router.post("/invitations/accept", response_model=AcceptedInvitationView)
async def accept(body: AcceptInput, user: VerifiedUser, db: Database):
    invitation = await db.scalar(
        select(Invitation).where(
            Invitation.token_hash == digest(body.token),
            Invitation.email == user.email,
            Invitation.revoked.is_(False),
            Invitation.consumed_at.is_(None),
            Invitation.expires_at > datetime.now(UTC),
        )
    )
    if invitation is None:
        raise HTTPException(400, "Приглашение недействительно или срок его действия истёк")
    await db.execute(
        text("SELECT set_config('app.organization_id', :id, true)"),
        {"id": str(invitation.organization_id)},
    )
    org = await db.scalar(
        select(Organization).where(Organization.id == invitation.organization_id).with_for_update()
    )
    if org is None or org.deletion_requested_at is not None:
        raise HTTPException(400, "Приглашение недействительно или срок его действия истёк")
    invitation = await db.scalar(
        select(Invitation)
        .where(
            Invitation.token_hash == digest(body.token),
            Invitation.email == user.email,
            Invitation.revoked.is_(False),
            Invitation.consumed_at.is_(None),
            Invitation.expires_at > datetime.now(UTC),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if invitation is None:
        raise HTTPException(400, "Приглашение недействительно или срок его действия истёк")
    member = await db.get(Membership, (invitation.organization_id, user.id))
    if member is None:
        member = Membership(
            organization_id=invitation.organization_id, user_id=user.id, role=invitation.role
        )
        db.add(member)
    invitation.consumed_at = datetime.now(UTC)
    audit(db, invitation.organization_id, user, "invitation.accepted", invitation.id)
    return {"organization_id": str(invitation.organization_id), "role": member.role}


@router.post("/organizations/{org_id}/invitations/{invitation_id}/resend", status_code=202)
async def resend_invitation(
    org_id: UUID, invitation_id: UUID, request: Request, user: VerifiedUser, db: Database
):
    actor = await organization_context(db, user, org_id, "manage_members", lock=True)
    invitation = await db.get(Invitation, invitation_id)
    if invitation is None:
        raise HTTPException(404, "Приглашение не найдено")
    if invitation.revoked or invitation.consumed_at is not None:
        raise HTTPException(409, "Приглашение уже отозвано или принято")
    if "manage_admins" not in ROLE_PERMISSIONS[actor.role] and invitation.role == "admin":
        raise HTTPException(403, "Управление администраторами доступно владельцу")
    await rate_limit(db, request, "invitation", f"{org_id}:{invitation.email}", 3, 60)
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(404, "Пространство не найдено")
    await send_invitation(db, request, invitation, org)
    audit(db, org_id, user, "invitation.resent", invitation_id)
    return {"message": "Письмо с приглашением поставлено в очередь"}
