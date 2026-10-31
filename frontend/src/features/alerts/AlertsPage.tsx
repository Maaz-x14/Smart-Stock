import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../lib/api";
import type { Alert } from "../../lib/types";
import styles from "./AlertsPage.module.css";

function useAlerts() {
  return useQuery({
    queryKey: ["alerts"],
    queryFn: () => api.get<{ alerts: Alert[] }>("/alerts"),
  });
}

function useDismiss() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (alertId: string) => api.patch(`/alerts/${alertId}/dismiss`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["alerts"] }),
  });
}

export function AlertsPage() {
  const { data, isLoading } = useAlerts();
  const dismiss = useDismiss();

  if (isLoading) return <p className={styles.status}>Loading alerts…</p>;

  const alerts = data?.alerts ?? [];

  return (
    <div className={styles.page}>
      <h1>Alerts</h1>
      {alerts.length === 0 ? (
        <p className={styles.status}>Nothing needs attention right now.</p>
      ) : (
        <ul className={styles.list}>
          {alerts.map((alert) => (
            <li key={alert.id} className={styles.alert}>
              <div className={styles.alertHeader}>
                <span>{new Date(alert.triggered_at).toLocaleDateString()}</span>
                <button type="button" onClick={() => dismiss.mutate(alert.id)}>
                  Dismiss
                </button>
              </div>
              <ul className={styles.items}>
                {alert.at_risk_items.map((item) => (
                  <li key={item.item_id}>
                    {item.name} — {item.days_remaining <= 0 ? "expired" : `${item.days_remaining} days left`}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
