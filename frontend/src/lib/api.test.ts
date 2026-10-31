import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { useAuthStore } from "./authStore";

describe("api client", () => {
  beforeEach(() => {
    useAuthStore.getState().clearSession();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("attaches the Authorization header when a token is set", async () => {
    useAuthStore.getState().setSession("test-token", null);
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await api.get("/inventory");

    const [, options] = fetchMock.mock.calls[0];
    expect((options.headers as Headers).get("Authorization")).toBe("Bearer test-token");
  });

  it("throws an ApiError with the server's code/message on a non-2xx response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ error: { code: "NOT_FOUND", message: "Item not found.", status: 404 } }),
        { status: 404 }
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.get("/inventory/unknown")).rejects.toMatchObject(
      new ApiError({ code: "NOT_FOUND", message: "Item not found.", status: 404 })
    );
  });

  it("retries once after a successful silent refresh on 401", async () => {
    useAuthStore.getState().setSession("expired-token", null);

    const fetchMock = vi
      .fn()
      // first call: the real request, comes back 401
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      // second call: /auth/refresh, succeeds
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ access_token: "new-token", token_type: "bearer", expires_in: 1800 }), {
          status: 200,
        })
      )
      // third call: the retried request, succeeds
      .mockResolvedValueOnce(new Response(JSON.stringify({ items: [] }), { status: 200 }));

    vi.stubGlobal("fetch", fetchMock);

    const result = await api.get("/inventory");

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(result).toEqual({ items: [] });
    expect(useAuthStore.getState().accessToken).toBe("new-token");
  });
});
