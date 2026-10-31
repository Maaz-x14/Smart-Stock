import { useMutation } from "@tanstack/react-query";
import { api } from "../../lib/api";
import type { ReceiptUploadResponse, StorageContext } from "../../lib/types";

export function useUploadReceipt() {
  return useMutation({
    mutationFn: async ({ file, storageContext }: { file: File; storageContext: StorageContext }) => {
      const form = new FormData();
      form.append("file", file);
      form.append("storage_context", storageContext);
      return api.postForm<ReceiptUploadResponse>("/receipts/upload", form);
    },
  });
}
