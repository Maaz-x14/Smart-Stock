import { create } from "zustand";

// Access token lives in memory only, never localStorage - a page
// reload loses it and the app calls /auth/refresh (the HttpOnly
// cookie survives a reload) to get a new one. See api.ts's
// ensureFreshToken.
interface AuthState {
  accessToken: string | null;
  user: { id: string; email: string; name: string } | null;
  setSession: (token: string, user: AuthState["user"]) => void;
  clearSession: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  accessToken: null,
  user: null,
  setSession: (accessToken, user) => set({ accessToken, user }),
  clearSession: () => set({ accessToken: null, user: null }),
}));
