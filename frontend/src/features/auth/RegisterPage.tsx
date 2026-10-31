import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError } from "../../lib/api";
import { useLogin, useRegister } from "./useAuth";
import styles from "./AuthPage.module.css";

export function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const register = useRegister();
  const login = useLogin();
  const navigate = useNavigate();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    try {
      await register.mutateAsync({ email, password, name });
      await login.mutateAsync({ email, password });
      navigate("/");
    } catch {
      // error is read from register.error/login.error below
    }
  }

  const error = register.error ?? login.error;

  return (
    <div className={styles.page}>
      <h1>Create your account</h1>
      <form onSubmit={handleSubmit} className={styles.form}>
        <label>
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} required autoFocus />
        </label>
        <label>
          Email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={8}
            required
          />
        </label>
        {error && (
          <p role="alert" className={styles.error}>
            {error instanceof ApiError ? error.message : "Something went wrong. Try again."}
          </p>
        )}
        <button type="submit" disabled={register.isPending || login.isPending}>
          {register.isPending || login.isPending ? "Creating account…" : "Create account"}
        </button>
      </form>
      <p className={styles.switchLink}>
        Already have an account? <a href="/login">Sign in</a>
      </p>
    </div>
  );
}
