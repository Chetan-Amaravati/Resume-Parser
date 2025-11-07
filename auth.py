import streamlit as st
import hashlib

def make_hash(password: str) -> str:
    """Hash password for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def show_login_register_page(db):
    st.title("🔐 Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # ---------- LOGIN ----------
    with tab1:
        st.subheader("Login to Continue")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            users_col = db.db["users"]
            user = users_col.find_one({"username": username, "password": make_hash(password)})
            if user:
                st.session_state["logged_in"] = True
                st.session_state["user"] = {
                    "username": username,
                    "email": user.get("email", "")
                }
                st.success(f"✅ Welcome {username}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

    # ---------- REGISTER ----------
    with tab2:
        st.subheader("Create a New Account")
        username = st.text_input("New Username", key="reg_user")
        email = st.text_input("Email (Gmail only)", key="reg_email", placeholder="example@gmail.com")
        password = st.text_input("Password", type="password", key="reg_pass")
        confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")

        if st.button("Register"):
            if not username or not password or not email:
                st.error("All fields are required.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif "@gmail.com" not in email:
                st.error("Please register using a Gmail address.")
            else:
                users_col = db.db["users"]
                if users_col.find_one({"username": username}):
                    st.warning("⚠️ Username already exists.")
                else:
                    users_col.insert_one({
                        "username": username,
                        "email": email,
                        "password": make_hash(password)
                    })
                    st.success("✅ Registration successful! You can now log in.")
