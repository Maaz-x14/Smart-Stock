import { Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/AppLayout";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { LoginPage } from "./features/auth/LoginPage";
import { RegisterPage } from "./features/auth/RegisterPage";
import { FridgePage } from "./features/fridge/FridgePage";
import { UploadPage } from "./features/receipts/UploadPage";
import { AlertsPage } from "./features/alerts/AlertsPage";
import { RecipesPage } from "./features/recipes/RecipesPage";
import { WastePage } from "./features/waste/WastePage";

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route element={<ProtectedRoute />}>
        <Route element={<AppLayout />}>
          <Route path="/" element={<FridgePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/recipes" element={<RecipesPage />} />
          <Route path="/waste" element={<WastePage />} />
        </Route>
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
