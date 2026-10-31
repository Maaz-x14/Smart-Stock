import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { UrgencyBadge } from "./UrgencyBadge";

describe("UrgencyBadge", () => {
  it("shows the day count for green/yellow/red tiers", () => {
    render(<UrgencyBadge tier="green" daysRemaining={6} />);
    expect(screen.getByText(/6 days left/)).toBeInTheDocument();
  });

  it("shows 'Expired' with no day count for the expired tier", () => {
    render(<UrgencyBadge tier="expired" daysRemaining={-2} />);
    expect(screen.getByText("Expired")).toBeInTheDocument();
    expect(screen.queryByText(/-2/)).not.toBeInTheDocument();
  });

  it("renders a distinct shape per tier so color is never the only signal", () => {
    const { container: green } = render(<UrgencyBadge tier="green" daysRemaining={6} />);
    const { container: red } = render(<UrgencyBadge tier="red" daysRemaining={1} />);
    const greenShape = green.querySelector("[aria-hidden='true']")?.textContent;
    const redShape = red.querySelector("[aria-hidden='true']")?.textContent;
    expect(greenShape).not.toEqual(redShape);
  });
});
