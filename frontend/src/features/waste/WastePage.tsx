import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../../lib/api";
import type { WasteLogResponse } from "../../lib/types";
import styles from "./WastePage.module.css";

function useWasteLog() {
  return useQuery({
    queryKey: ["waste-log"],
    queryFn: () => api.get<WasteLogResponse>("/waste-log"),
  });
}

export function WastePage() {
  const { data, isLoading } = useWasteLog();

  if (isLoading) return <p className={styles.status}>Loading…</p>;
  if (!data) return null;

  const chartData = [
    { label: "Consumed", count: data.summary.consumed },
    { label: "Wasted", count: data.summary.wasted },
  ];

  return (
    <div className={styles.page}>
      <h1>Waste tracker</h1>

      <div className={styles.summaryRow}>
        <div className={styles.stat}>
          <span className={styles.statValue}>{data.summary.total_items}</span>
          <span className={styles.statLabel}>items tracked</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statValue}>{data.summary.waste_rate_percent}%</span>
          <span className={styles.statLabel}>waste rate</span>
        </div>
      </div>

      {data.summary.total_items > 0 && (
        <div className={styles.chartWrap}>
          <BarChart width={320} height={200} data={chartData}>
            <CartesianGrid vertical={false} stroke="var(--color-border)" />
            <XAxis dataKey="label" tick={{ fill: "var(--color-text-muted)", fontSize: 13 }} />
            <YAxis tick={{ fill: "var(--color-text-muted)", fontSize: 13 }} allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="var(--color-compressor)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </div>
      )}

      <h2 className={styles.logTitle}>Recent activity</h2>
      <ul className={styles.log}>
        {data.entries.map((entry) => (
          <li key={entry.id} className={styles.logRow}>
            <span>{entry.item_name}</span>
            <span className={entry.outcome === "WASTED" ? styles.wasted : styles.consumed}>
              {entry.outcome.toLowerCase()}
            </span>
            <span className={styles.logDate}>{new Date(entry.recorded_at).toLocaleDateString()}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
