import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError } from "../../lib/api";
import { useLogin } from "./useAuth";
import styles from "./AuthPage.module.css";

export function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const login = useLogin();
  const navigate = useNavigate();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    try {
      await login.mutateAsync({ email, password });
      navigate("/");
    } catch {
      // error is read from login.error below; nothing further to do
    }
  }

  return (
    <div className={styles.page}>
      <h1>Smart-Stock</h1>
      <form onSubmit={handleSubmit} className={styles.form}>
        <label>
          Email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        {login.isError && (
          <p role="alert" className={styles.error}>
            {login.error instanceof ApiError ? login.error.message : "Something went wrong. Try again."}
          </p>
        )}
        <button type="submit" disabled={login.isPending}>
          {login.isPending ? "Signing in…" : "Sign in"}
        </button>
      </form>
      <p className={styles.switchLink}>
        New here? <a href="/register">Create an account</a>
      </p>
    </div>
  );
}
