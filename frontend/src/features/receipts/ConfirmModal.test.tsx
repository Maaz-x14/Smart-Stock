import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConfirmModal } from "./ConfirmModal";
import type { ExtractedItem } from "../../lib/types";

const foodItem: ExtractedItem = {
  raw_token: "ORG STRWBRY 1LB",
  brand: null,
  is_food: true,
  unit: "lb",
  canonical_name: "Strawberries",
  quantity: 1,
  category: "Produce",
  predicted_expiry_date: "2026-11-05",
  shelf_life_days: 7,
  confidence: 0.91,
  storage_context: "fridge",
  normalization_pass: 1,
};

const nonFoodItem: ExtractedItem = {
  raw_token: "Supravit-M Tablet 10's",
  brand: "Supravit",
  is_food: false,
  unit: null,
  canonical_name: null,
  quantity: 6,
  category: null,
  predicted_expiry_date: null,
  shelf_life_days: null,
  confidence: null,
  storage_context: null,
  normalization_pass: null,
};

describe("ConfirmModal", () => {
  it("starts non-food items unchecked and food items checked", () => {
    render(
      <ConfirmModal
        items={[foodItem, nonFoodItem]}
        defaultStorageContext="fridge"
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        isSaving={false}
      />
    );

    const checkboxes = screen.getAllByRole("checkbox") as HTMLInputElement[];
    expect(checkboxes[0].checked).toBe(true); // Strawberries (is_food=true)
    expect(checkboxes[1].checked).toBe(false); // Supravit tablet (is_food=false)
  });

  it("flags the non-food item as excluded", () => {
    render(
      <ConfirmModal
        items={[nonFoodItem]}
        defaultStorageContext="fridge"
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        isSaving={false}
      />
    );
    expect(screen.getByText(/not a food item/i)).toBeInTheDocument();
  });

  it("only submits checked items, mapped to ConfirmItem shape", async () => {
    const onConfirm = vi.fn();
    const user = userEvent.setup();
    render(
      <ConfirmModal
        items={[foodItem, nonFoodItem]}
        defaultStorageContext="fridge"
        onConfirm={onConfirm}
        onCancel={vi.fn()}
        isSaving={false}
      />
    );

    await user.click(screen.getByRole("button", { name: /add 1 item/i }));

    expect(onConfirm).toHaveBeenCalledWith([
      {
        canonical_name: "Strawberries",
        brand: null,
        quantity: 1,
        unit: "lb",
        category: "Produce",
        predicted_expiry_date: "2026-11-05",
        storage_context: "fridge",
      },
    ]);
  });

  it("closes on Escape", async () => {
    const onCancel = vi.fn();
    const user = userEvent.setup();
    render(
      <ConfirmModal
        items={[foodItem]}
        defaultStorageContext="fridge"
        onConfirm={vi.fn()}
        onCancel={onCancel}
        isSaving={false}
      />
    );
    await user.keyboard("{Escape}");
    expect(onCancel).toHaveBeenCalled();
  });

  it("disables the confirm button while saving", () => {
    render(
      <ConfirmModal
        items={[foodItem]}
        defaultStorageContext="fridge"
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        isSaving={true}
      />
    );
    expect(screen.getByRole("button", { name: /saving/i })).toBeDisabled();
  });
});
