import { useMemo, useState } from "react";
import { ItemTag } from "../../components/ItemTag";
import type { Category, InventoryItemDTO } from "../../lib/types";
import { useInventory, useMarkUsed } from "./useInventory";
import styles from "./FridgePage.module.css";

const CATEGORY_ORDER: Category[] = ["Produce", "Dairy", "Meat", "Pantry", "Frozen"];

function groupByCategory(items: InventoryItemDTO[]) {
  const groups = new Map<string, InventoryItemDTO[]>();
  for (const item of items) {
    const list = groups.get(item.category) ?? [];
    list.push(item);
    groups.set(item.category, list);
  }
  for (const list of groups.values()) {
    list.sort((a, b) => a.days_remaining - b.days_remaining);
  }
  return groups;
}

export function FridgePage() {
  const { data, isLoading, isError } = useInventory();
  const markUsed = useMarkUsed();
  const [removingIds, setRemovingIds] = useState<Set<string>>(new Set());

  const groups = useMemo(() => groupByCategory(data?.items ?? []), [data]);
  const useSoonCount = (data?.items ?? []).filter((i) => i.urgency_tier === "red" || i.urgency_tier === "expired").length;

  function handleMarkUsed(itemId: string) {
    // Optimistic exit animation (design plan: "an item you mark used
    // slides off its shelf"), then the real mutation.
    setRemovingIds((prev) => new Set(prev).add(itemId));
    setTimeout(() => markUsed.mutate(itemId), 200);
  }

  if (isLoading) {
    return <p className={styles.status}>Loading your fridge…</p>;
  }

  if (isError) {
    return (
      <p className={styles.status} role="alert">
        Couldn't load your fridge. Check your connection and try again.
      </p>
    );
  }

  const orderedCategories = [
    ...CATEGORY_ORDER.filter((c) => groups.has(c)),
    ...[...groups.keys()].filter((c) => !CATEGORY_ORDER.includes(c as Category)),
  ];

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1>Your fridge</h1>
        {useSoonCount > 0 && (
          <p className={styles.summary}>
            Use soon: {useSoonCount} {useSoonCount === 1 ? "item" : "items"}
          </p>
        )}
      </div>

      {orderedCategories.length === 0 ? (
        <p className={styles.status}>
          Nothing in your fridge yet. Upload a receipt to get started.
        </p>
      ) : (
        orderedCategories.map((category) => (
          <section key={category} className={styles.shelf}>
            <h2 className={styles.shelfName}>{category}</h2>
            <ul className={styles.tagList}>
              {groups.get(category)!.map((item) => (
                <ItemTag
                  key={item.id}
                  item={item}
                  onMarkUsed={handleMarkUsed}
                  isRemoving={removingIds.has(item.id)}
                />
              ))}
            </ul>
          </section>
        ))
      )}
    </div>
  );
}
