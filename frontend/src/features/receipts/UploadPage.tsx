import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../../lib/api";
import type { ConfirmItem, ConfirmResponse, ExtractedItem, StorageContext } from "../../lib/types";
import { INVENTORY_QUERY_KEY } from "../fridge/useInventory";
import { ScanProgress } from "./ScanProgress";
import { ConfirmModal } from "./ConfirmModal";
import { useUploadReceipt } from "./useUpload";
import styles from "./UploadPage.module.css";

export function UploadPage() {
  const [storageContext, setStorageContext] = useState<StorageContext>("fridge");
  const [receiptId, setReceiptId] = useState<string | null>(null);
  const [extractedItems, setExtractedItems] = useState<ExtractedItem[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const upload = useUploadReceipt();
  const queryClient = useQueryClient();

  const confirm = useMutation({
    mutationFn: (items: ConfirmItem[]) =>
      api.post<ConfirmResponse>("/receipts/confirm", { receipt_id: receiptId, items }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: INVENTORY_QUERY_KEY });
      setExtractedItems(null);
      setReceiptId(null);
    },
  });

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    upload.mutate(
      { file, storageContext },
      {
        onSuccess: (response) => {
          setReceiptId(response.receipt_id);
          setExtractedItems(response.extracted_items);
        },
      }
    );
  }

  return (
    <div className={styles.page}>
      <h1>Add a receipt</h1>
      <p className={styles.hint}>A clear, flat photo works best. This takes about 10-15 seconds.</p>

      <label className={styles.storageLabel}>
        Default storage
        <select
          value={storageContext}
          onChange={(e) => setStorageContext(e.target.value as StorageContext)}
          disabled={upload.isPending}
        >
          <option value="fridge">Fridge</option>
          <option value="freezer">Freezer</option>
          <option value="pantry">Pantry</option>
        </select>
      </label>

      {upload.isPending ? (
        <ScanProgress />
      ) : (
        <button
          type="button"
          className={styles.uploadButton}
          onClick={() => fileInputRef.current?.click()}
        >
          Choose a photo
        </button>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,application/pdf"
        onChange={handleFileChange}
        className={styles.hiddenInput}
      />

      {upload.isError && (
        <p role="alert" className={styles.error}>
          We couldn't read this receipt. Try again in better light, or hold the camera flatter.
        </p>
      )}

      {extractedItems && (
        <ConfirmModal
          items={extractedItems}
          defaultStorageContext={storageContext}
          isSaving={confirm.isPending}
          onCancel={() => {
            setExtractedItems(null);
            setReceiptId(null);
          }}
          onConfirm={(items) => confirm.mutate(items)}
        />
      )}
    </div>
  );
}
