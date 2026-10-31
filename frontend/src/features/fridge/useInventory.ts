import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../lib/api";
import type { InventoryListResponse } from "../../lib/types";

export const INVENTORY_QUERY_KEY = ["inventory"] as const;

export function useInventory() {
  return useQuery({
    queryKey: INVENTORY_QUERY_KEY,
    queryFn: () => api.get<InventoryListResponse>("/inventory?limit=100"),
  });
}

interface BulkConsumeResponse {
  consumed_count: number;
  waste_log_ids: string[];
}

export function useMarkUsed() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (itemId: string) =>
      api.post<BulkConsumeResponse>("/inventory/bulk-consume", { item_ids: [itemId] }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: INVENTORY_QUERY_KEY });
    },
  });
}
