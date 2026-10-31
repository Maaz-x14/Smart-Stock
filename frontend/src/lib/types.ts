// Hand-written to mirror API_Spec.md v1.1 exactly. The plan called for
// types generated from FastAPI's OpenAPI schema - that step needs the
// backend actually running to introspect, which this build environment
// doesn't have wired up yet. These are written to the same contract the
// backend tests assert against (tests/test_*.py), and should be
// regenerated from OpenAPI once there's a running backend to point at;
// see HANDOFF.md.

export type UrgencyTier = "green" | "yellow" | "red" | "expired";
export type StorageContext = "fridge" | "freezer" | "pantry";
export type ItemStatus = "ACTIVE" | "CONSUMED" | "WASTED";
export type Category = "Produce" | "Dairy" | "Meat" | "Pantry" | "Frozen";

export interface User {
  id: string;
  email: string;
  name: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface ExtractedItem {
  raw_token: string;
  brand: string | null;
  is_food: boolean | null;
  unit: string | null;
  canonical_name: string | null;
  quantity: number | null;
  category: string | null;
  predicted_expiry_date: string | null;
  shelf_life_days: number | null;
  confidence: number | null;
  storage_context: StorageContext | null;
  normalization_pass: number | null;
}

export interface ReceiptUploadResponse {
  receipt_id: string;
  extracted_items: ExtractedItem[];
  total_items_extracted: number;
  processing_time_ms: number;
}

export interface ConfirmItem {
  canonical_name: string;
  brand: string | null;
  quantity: number;
  unit: string | null;
  category: string;
  predicted_expiry_date: string;
  storage_context: StorageContext;
}

export interface ConfirmResponse {
  created_items: number;
  inventory_ids: string[];
}

export interface InventoryItemDTO {
  id: string;
  canonical_name: string;
  brand: string | null;
  quantity: number;
  unit: string | null;
  category: string;
  purchase_date: string;
  predicted_expiry_date: string;
  days_remaining: number;
  urgency_tier: UrgencyTier;
  storage_context: StorageContext;
  status: ItemStatus;
  created_at: string;
}

export interface InventoryListResponse {
  items: InventoryItemDTO[];
  total: number;
  page: number;
  limit: number;
}

export interface AtRiskItem {
  item_id: string;
  name: string;
  expiry_date: string;
  days_remaining: number;
}

export interface Alert {
  id: string;
  triggered_at: string;
  expiry_threshold_hours: number;
  status: "ACTIVE" | "DISMISSED";
  at_risk_items: AtRiskItem[];
}

export interface RecipeResult {
  id: string;
  title: string;
  image_url: string | null;
  ready_in_minutes: number | null;
  servings: number | null;
  source_url: string | null;
  matched_ingredients: string[];
  matched_count: number;
  missing_ingredients: string[];
}

export interface WasteLogEntry {
  id: string;
  item_name: string;
  outcome: "CONSUMED" | "WASTED";
  recorded_at: string;
  recipe_used: string | null;
}

export interface WasteLogResponse {
  summary: {
    total_items: number;
    consumed: number;
    wasted: number;
    waste_rate_percent: number;
  };
  entries: WasteLogEntry[];
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    status: number;
  };
}
