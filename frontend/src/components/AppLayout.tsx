import { NavLink, Outlet } from "react-router-dom";
import { OfflineBanner } from "./OfflineBanner";
import { useLogout } from "../features/auth/useAuth";
import styles from "./AppLayout.module.css";

const NAV_ITEMS = [
  { to: "/", label: "Fridge" },
  { to: "/alerts", label: "Alerts" },
  { to: "/recipes", label: "Recipes" },
  { to: "/waste", label: "Waste" },
];

export function AppLayout() {
  const logout = useLogout();

  return (
    <div className={styles.shell}>
      <OfflineBanner />
      <nav className={styles.sidebar} aria-label="Main">
        <span className={styles.brand}>Smart-Stock</span>
        <ul className={styles.navList}>
          {NAV_ITEMS.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) => (isActive ? styles.navLinkActive : styles.navLink)}
              >
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
        <NavLink to="/upload" className={styles.addButton}>
          Add receipt
        </NavLink>
        <button type="button" className={styles.logoutButton} onClick={() => logout.mutate()}>
          Sign out
        </button>
      </nav>

      <main className={styles.content}>
        <Outlet />
      </main>

      <nav className={styles.bottomBar} aria-label="Main">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => (isActive ? styles.bottomLinkActive : styles.bottomLink)}
          >
            {item.label}
          </NavLink>
        ))}
        <NavLink to="/upload" className={styles.bottomLink}>
          Add
        </NavLink>
      </nav>
    </div>
  );
}
