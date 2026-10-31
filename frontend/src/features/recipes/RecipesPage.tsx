import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { useInventory } from "../fridge/useInventory";
import type { RecipeResult } from "../../lib/types";
import styles from "./RecipesPage.module.css";

function useRecipes(ingredients: string[]) {
  return useQuery({
    queryKey: ["recipes", ingredients],
    queryFn: () =>
      api.get<{ recipes: RecipeResult[]; cached: boolean }>(
        `/recipes?ingredients=${encodeURIComponent(ingredients.join(","))}`
      ),
    enabled: ingredients.length > 0,
  });
}

export function RecipesPage() {
  const { data: inventory } = useInventory();
  const soonestItems = (inventory?.items ?? [])
    .filter((i) => i.urgency_tier === "red" || i.urgency_tier === "yellow")
    .slice(0, 5)
    .map((i) => i.canonical_name);

  const { data, isLoading, isError } = useRecipes(soonestItems);

  return (
    <div className={styles.page}>
      <h1>Recipes</h1>
      <p className={styles.hint}>Based on what's about to expire.</p>

      {soonestItems.length === 0 && <p className={styles.status}>Nothing close to expiring right now.</p>}
      {isLoading && <p className={styles.status}>Looking for recipes…</p>}
      {isError && (
        <p className={styles.status} role="alert">
          Recipe suggestions aren't available right now. Try again later.
        </p>
      )}

      <ul className={styles.list}>
        {data?.recipes.map((recipe) => (
          <li key={recipe.id} className={styles.card}>
            <h2 className={styles.title}>{recipe.title}</h2>
            <p className={styles.meta}>
              Uses: {recipe.matched_ingredients.join(", ")}
              {recipe.missing_ingredients.length > 0 && ` · Needs: ${recipe.missing_ingredients.join(", ")}`}
            </p>
            {recipe.source_url && (
              <a href={recipe.source_url} target="_blank" rel="noreferrer">
                View recipe
              </a>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
