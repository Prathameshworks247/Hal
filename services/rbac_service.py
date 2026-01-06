"""
Role-Based Access Control (RBAC) service for document access and query filtering.
Provides user authentication, role management, and document-level permissions.
"""
import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"  # Full access: read, write, delete, manage users
    EDITOR = "editor"  # Read and write access to documents
    VIEWER = "viewer"  # Read-only access to documents
    GUEST = "guest"  # Limited read access


class Permission(Enum):
    """Granular permissions."""
    READ_DOCUMENT = "read_document"
    WRITE_DOCUMENT = "write_document"
    DELETE_DOCUMENT = "delete_document"
    QUERY_SYSTEM = "query_system"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: {
        Permission.READ_DOCUMENT,
        Permission.WRITE_DOCUMENT,
        Permission.DELETE_DOCUMENT,
        Permission.QUERY_SYSTEM,
        Permission.MANAGE_USERS,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_DATA
    },
    Role.EDITOR: {
        Permission.READ_DOCUMENT,
        Permission.WRITE_DOCUMENT,
        Permission.QUERY_SYSTEM,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_DATA
    },
    Role.VIEWER: {
        Permission.READ_DOCUMENT,
        Permission.QUERY_SYSTEM,
        Permission.VIEW_ANALYTICS
    },
    Role.GUEST: {
        Permission.READ_DOCUMENT,
        Permission.QUERY_SYSTEM
    }
}


class User:
    """User model with authentication and authorization."""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        role: Role,
        password_hash: Optional[str] = None,
        allowed_documents: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.password_hash = password_hash
        self.allowed_documents = allowed_documents or set()  # Empty = all documents
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.last_login = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())
    
    def can_access_document(self, document_id: str) -> bool:
        """Check if user can access a specific document."""
        # If allowed_documents is empty, user can access all documents
        if not self.allowed_documents:
            return True
        return document_id in self.allowed_documents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (without password hash)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role.value,
            "allowed_documents": list(self.allowed_documents),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_login": self.last_login
        }


class RBACManager:
    """Manages users, roles, and permissions."""
    
    def __init__(self, storage_path: str = "rbac_data"):
        """
        Initialize RBAC manager.
        
        Args:
            storage_path: Directory to store user data
        """
        self.storage_path = storage_path
        self.users_file = os.path.join(storage_path, "users.json")
        self.audit_log_file = os.path.join(storage_path, "audit_log.json")
        
        os.makedirs(storage_path, exist_ok=True)
        
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, str] = {}  # session_token -> user_id
        
        self._load_users()
        self._create_default_admin()
    
    def _load_users(self):
        """Load users from storage."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data:
                        user = User(
                            user_id=user_data["user_id"],
                            username=user_data["username"],
                            role=Role(user_data["role"]),
                            password_hash=user_data.get("password_hash"),
                            allowed_documents=set(user_data.get("allowed_documents", [])),
                            metadata=user_data.get("metadata", {})
                        )
                        user.created_at = user_data.get("created_at", user.created_at)
                        user.last_login = user_data.get("last_login")
                        self.users[user.user_id] = user
                logger.info(f"Loaded {len(self.users)} users")
            except Exception as e:
                logger.error(f"Error loading users: {e}")
    
    def _save_users(self):
        """Save users to storage."""
        try:
            data = []
            for user in self.users.values():
                user_dict = user.to_dict()
                user_dict["password_hash"] = user.password_hash
                data.append(user_dict)
            
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _create_default_admin(self):
        """Create default admin user if no users exist."""
        if not self.users:
            admin_password = "admin123"  # Change this in production!
            admin_user = self.create_user(
                username="admin",
                password=admin_password,
                role=Role.ADMIN
            )
            logger.warning(f"Created default admin user. Username: admin, Password: {admin_password}")
            logger.warning("IMPORTANT: Change the admin password immediately!")
    
    def create_user(
        self,
        username: str,
        password: str,
        role: Role,
        allowed_documents: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            password: Plain text password (will be hashed)
            role: User role
            allowed_documents: Optional set of allowed document IDs
            metadata: Optional user metadata
        
        Returns:
            Created User object
        """
        user_id = hashlib.sha256(f"{username}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        password_hash = self._hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            password_hash=password_hash,
            allowed_documents=allowed_documents,
            metadata=metadata
        )
        
        self.users[user_id] = user
        self._save_users()
        
        self._log_audit_event("user_created", user_id, {"username": username, "role": role.value})
        logger.info(f"Created user: {username} (ID: {user_id}, Role: {role.value})")
        
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and create session.
        
        Args:
            username: Username
            password: Password
        
        Returns:
            Session token if successful, None otherwise
        """
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            logger.warning(f"Authentication failed: user not found - {username}")
            return None
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            logger.warning(f"Authentication failed: invalid password - {username}")
            return None
        
        # Create session
        session_token = secrets.token_urlsafe(32)
        self.sessions[session_token] = user.user_id
        
        # Update last login
        user.last_login = datetime.now().isoformat()
        self._save_users()
        
        self._log_audit_event("user_login", user.user_id, {"username": username})
        logger.info(f"User authenticated: {username}")
        
        return session_token
    
    def get_user_from_session(self, session_token: str) -> Optional[User]:
        """Get user from session token."""
        user_id = self.sessions.get(session_token)
        if user_id:
            return self.users.get(user_id)
        return None
    
    def logout(self, session_token: str):
        """Logout user and invalidate session."""
        if session_token in self.sessions:
            user_id = self.sessions[session_token]
            del self.sessions[session_token]
            self._log_audit_event("user_logout", user_id, {})
            logger.info(f"User logged out: {user_id}")
    
    def check_permission(self, session_token: str, permission: Permission) -> bool:
        """Check if user has permission."""
        user = self.get_user_from_session(session_token)
        if not user:
            return False
        return user.has_permission(permission)
    
    def filter_documents_by_access(
        self,
        session_token: str,
        documents: List[Any]
    ) -> List[Any]:
        """
        Filter documents based on user access rights.
        
        Args:
            session_token: User session token
            documents: List of documents to filter
        
        Returns:
            Filtered list of documents user can access
        """
        user = self.get_user_from_session(session_token)
        if not user:
            return []
        
        # Admin and users with no restrictions see all documents
        if user.role == Role.ADMIN or not user.allowed_documents:
            return documents
        
        # Filter based on allowed documents
        filtered = []
        for doc in documents:
            doc_id = self._get_document_id(doc)
            if user.can_access_document(doc_id):
                filtered.append(doc)
        
        return filtered
    
    def _get_document_id(self, document: Any) -> str:
        """Extract document ID from document object."""
        if hasattr(document, 'metadata'):
            return document.metadata.get('source', '') or document.metadata.get('file_path', '')
        return str(document)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 (use bcrypt in production)."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return self._hash_password(password) == password_hash
    
    def _log_audit_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log audit event."""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "user_id": user_id,
                "details": details
            }
            
            # Append to audit log
            audit_log = []
            if os.path.exists(self.audit_log_file):
                with open(self.audit_log_file, 'r') as f:
                    audit_log = json.load(f)
            
            audit_log.append(event)
            
            # Keep only last 1000 events
            if len(audit_log) > 1000:
                audit_log = audit_log[-1000:]
            
            with open(self.audit_log_file, 'w') as f:
                json.dump(audit_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        try:
            if os.path.exists(self.audit_log_file):
                with open(self.audit_log_file, 'r') as f:
                    audit_log = json.load(f)
                return audit_log[-limit:]
        except Exception as e:
            logger.error(f"Error reading audit log: {e}")
        return []


# Global RBAC manager instance
_rbac_manager = None

def get_rbac_manager() -> RBACManager:
    """Get or create global RBAC manager instance."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager

