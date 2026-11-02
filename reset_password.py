"""
Owner-only account recovery. Run on the machine/server where the app lives:

    python reset_password.py --list                 # show registered emails
    python reset_password.py you@email.com          # prompts for a new password
    python reset_password.py you@email.com NewPass  # non-interactive
"""
import getpass
import sys

from werkzeug.security import generate_password_hash

from app import app, db, User


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    with app.app_context():
        if args[0] == "--list":
            users = User.query.order_by(User.id).all()
            if not users:
                print("No accounts found.")
            for u in users:
                print(f"{u.id}: {u.email}  ({u.name})")
            return 0

        email = args[0].strip().lower()
        user = User.query.filter_by(email=email).first()
        if not user:
            print(f"No account with email {email}. Use --list to see registered emails.")
            return 1

        if len(args) > 1:
            password = args[1]
        else:
            password = getpass.getpass("New password: ")
            if password != getpass.getpass("Confirm new password: "):
                print("Passwords do not match.")
                return 1

        if len(password) < 6:
            print("Password must be at least 6 characters.")
            return 1

        user.password = generate_password_hash(password)
        db.session.commit()
        print(f"Password updated for {user.email}. You can log in now.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
