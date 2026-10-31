import type { InventoryItemDTO } from "../lib/types";
import { UrgencyBadge } from "./UrgencyBadge";
import styles from "./ItemTag.module.css";

interface ItemTagProps {
  item: InventoryItemDTO;
  onMarkUsed: (id: string) => void;
  isRemoving?: boolean;
}

// The shelf-edge price-tag concept from the design plan: a flat
// rectangle, one bold line of the item name, the quantity/unit in
// mono underneath, urgency on the right. Not a card - no shadow, no
// rounded-corner-card-kit look.
export function ItemTag({ item, onMarkUsed, isRemoving }: ItemTagProps) {
  return (
    <li className={`${styles.tag} ${isRemoving ? styles.removing : ""}`}>
      <div className={styles.main}>
        <span className={styles.name}>{item.canonical_name}</span>
        <span className={styles.meta}>
          {item.quantity} {item.unit ?? ""}
          {item.brand ? ` · ${item.brand}` : ""}
        </span>
      </div>
      <UrgencyBadge tier={item.urgency_tier} daysRemaining={item.days_remaining} />
      <button
        type="button"
        className={styles.useButton}
        onClick={() => onMarkUsed(item.id)}
        aria-label={`Mark ${item.canonical_name} as used`}
      >
        Used it
      </button>
    </li>
  );
}
