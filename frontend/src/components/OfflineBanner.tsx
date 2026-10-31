import { useEffect, useState } from "react";
import styles from "./OfflineBanner.module.css";

// HCI rule: visibility of system status. The upload flow especially
// needs this - a 12-15s wait with no network looks identical to a
// slow server unless the user is told which one it is.
export function OfflineBanner() {
  const [isOffline, setIsOffline] = useState(() => !navigator.onLine);

  useEffect(() => {
    const goOnline = () => setIsOffline(false);
    const goOffline = () => setIsOffline(true);
    window.addEventListener("online", goOnline);
    window.addEventListener("offline", goOffline);
    return () => {
      window.removeEventListener("online", goOnline);
      window.removeEventListener("offline", goOffline);
    };
  }, []);

  if (!isOffline) return null;

  return (
    <div role="status" className={styles.banner}>
      You're offline. Uploading a receipt and saving changes won't work until you're back online.
    </div>
  );
}
