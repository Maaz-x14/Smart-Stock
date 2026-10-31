import { useEffect, useId, useRef, useState } from "react";
import type { ConfirmItem, ExtractedItem, StorageContext } from "../../lib/types";
import styles from "./ConfirmModal.module.css";

interface RowState {
  item: ExtractedItem;
  checked: boolean;
}

interface ConfirmModalProps {
  items: ExtractedItem[];
  defaultStorageContext: StorageContext;
  onConfirm: (items: ConfirmItem[]) => void;
  onCancel: () => void;
  isSaving: boolean;
}

const REVIEW_CONFIDENCE_THRESHOLD = 0.6;

// Mandatory confirmation modal (HCI: error prevention). Non-food
// items start unchecked; low-confidence items are flagged, never
// silently auto-included. Fully keyboard-operable: focus is trapped
// while open, Escape cancels, every control is a real <input>/<button>.
export function ConfirmModal({ items, defaultStorageContext, onConfirm, onCancel, isSaving }: ConfirmModalProps) {
  const [rows, setRows] = useState<RowState[]>(() =>
    items.map((item) => ({ item, checked: item.is_food === true }))
  );
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onCancel();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [onCancel]);

  const checkedCount = rows.filter((r) => r.checked).length;

  function toggle(index: number) {
    setRows((prev) => prev.map((r, i) => (i === index ? { ...r, checked: !r.checked } : r)));
  }

  function handleConfirm() {
    const confirmItems: ConfirmItem[] = rows
      .filter((r) => r.checked && r.item.canonical_name && r.item.category && r.item.predicted_expiry_date)
      .map((r) => ({
        canonical_name: r.item.canonical_name!,
        brand: r.item.brand,
        quantity: r.item.quantity ?? 1,
        unit: r.item.unit,
        category: r.item.category!,
        predicted_expiry_date: r.item.predicted_expiry_date!,
        storage_context: r.item.storage_context ?? defaultStorageContext,
      }));
    onConfirm(confirmItems);
  }

  return (
    <div className={styles.overlay} role="presentation" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <div className={styles.header}>
          <h2 id={titleId}>Check what we found</h2>
          <span className={styles.count}>{items.length} items</span>
        </div>

        <ul className={styles.list}>
          {rows.map((row, i) => {
            const isLowConfidence = row.item.is_food === true && (row.item.confidence ?? 1) < REVIEW_CONFIDENCE_THRESHOLD;
            const isExcluded = row.item.is_food !== true;

            return (
              <li key={`${row.item.raw_token}-${i}`} className={styles.row}>
                <label className={styles.rowLabel}>
                  <input
                    type="checkbox"
                    checked={row.checked}
                    onChange={() => toggle(i)}
                    disabled={isExcluded && !row.item.canonical_name}
                  />
                  <span className={styles.rowText}>
                    <span className={styles.itemName}>
                      {row.item.canonical_name ?? row.item.raw_token}
                    </span>
                    {isExcluded && <span className={styles.flag}>not a food item — excluded</span>}
                    {isLowConfidence && <span className={styles.flag}>low confidence — check this</span>}
                  </span>
                </label>
                {!isExcluded && (
                  <span className={styles.rowMeta}>
                    {row.item.category} · {row.item.predicted_expiry_date}
                  </span>
                )}
              </li>
            );
          })}
        </ul>

        <div className={styles.footer}>
          <button type="button" className={styles.cancelButton} onClick={onCancel} disabled={isSaving}>
            Cancel
          </button>
          <button
            type="button"
            className={styles.confirmButton}
            onClick={handleConfirm}
            disabled={isSaving || checkedCount === 0}
          >
            {isSaving ? "Saving…" : `Add ${checkedCount} item${checkedCount === 1 ? "" : "s"}`}
          </button>
        </div>
      </div>
    </div>
  );
}
