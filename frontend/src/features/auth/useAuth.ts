import { useMutation } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { useAuthStore } from "../../lib/authStore";
import type { TokenResponse } from "../../lib/types";

export function useLogin() {
  const setSession = useAuthStore((s) => s.setSession);
  return useMutation({
    mutationFn: (payload: { email: string; password: string }) =>
      api.post<TokenResponse>("/auth/login", payload),
    onSuccess: (response, variables) => {
      setSession(response.access_token, { id: "", email: variables.email, name: "" });
    },
  });
}

export function useRegister() {
  return useMutation({
    mutationFn: (payload: { email: string; password: string; name: string }) =>
      api.post("/auth/register", payload),
  });
}

export function useLogout() {
  const clearSession = useAuthStore((s) => s.clearSession);
  return useMutation({
    mutationFn: () => api.post("/auth/logout"),
    onSuccess: () => clearSession(),
  });
}
