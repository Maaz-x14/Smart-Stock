import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import styles from "./ScanProgress.module.css";

// The one deliberate motion moment the design plan calls for. Upload
// genuinely takes ~12-15s (measured: OCR ~5.3s, Stage 2 ~5.6s - see
// plan doc "Risks"), so these stages are honest about elapsed time,
// not a fake progress bar: each stage holds roughly as long as its
// real pipeline stage does, and the component keeps showing the last
// stage for as long as the actual request takes if it runs over.
const STAGES = [
  { label: "Reading the receipt", durationMs: 6000 },
  { label: "Finding the items", durationMs: 6000 },
  { label: "Estimating expiry", durationMs: 3000 },
];

export function ScanProgress() {
  const [stageIndex, setStageIndex] = useState(0);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    if (stageIndex >= STAGES.length - 1) return;
    const timer = setTimeout(() => setStageIndex((i) => i + 1), STAGES[stageIndex].durationMs);
    return () => clearTimeout(timer);
  }, [stageIndex]);

  return (
    <div className={styles.container} role="status" aria-live="polite">
      <div className={styles.stages}>
        {STAGES.map((stage, i) => (
          <div key={stage.label} className={styles.stageRow}>
            <span className={styles.marker} aria-hidden="true">
              {i < stageIndex ? "✓" : i === stageIndex ? "●" : "○"}
            </span>
            <span className={i <= stageIndex ? styles.activeLabel : styles.label}>{stage.label}</span>
          </div>
        ))}
      </div>
      {!reduceMotion && (
        <motion.div
          className={styles.receiptStrip}
          initial={{ y: 0 }}
          animate={{ y: [0, -6, 0] }}
          transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
          aria-hidden="true"
        />
      )}
    </div>
  );
}
