import type { UrgencyTier } from "../lib/types";
import styles from "./UrgencyBadge.module.css";

// Urgency never relies on color alone (design plan, HCI rules /
// Accessibility): each tier carries its own word and its own shape,
// so it still reads correctly for color-blind users and in print.
const TIER_CONFIG: Record<UrgencyTier, { label: string; shape: string }> = {
  green: { label: "days left", shape: "●" },
  yellow: { label: "days left", shape: "▲" },
  red: { label: "days left", shape: "■" },
  expired: { label: "Expired", shape: "■" },
};

export function UrgencyBadge({ tier, daysRemaining }: { tier: UrgencyTier; daysRemaining: number }) {
  const config = TIER_CONFIG[tier];
  const text = tier === "expired" ? "Expired" : `${daysRemaining} ${config.label}`;

  return (
    <span className={`${styles.badge} ${styles[tier]}`}>
      <span aria-hidden="true" className={styles.shape}>
        {config.shape}
      </span>
      {text}
    </span>
  );
}
