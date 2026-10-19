import { describe, it, expect } from "vitest";
import { translations } from "../lib/i18n/translations";

describe("i18n Translations Dictionary", () => {
  it("has matching keys for English and Arabic", () => {
    const enKeys = Object.keys(translations.en).sort();
    const arKeys = Object.keys(translations.ar).sort();

    expect(enKeys).toEqual(arKeys);
  });

  it("has valid refusal reasons for both languages", () => {
    const reasons = ["out_of_scope", "no_evidence", "unsafe_request", "policy"] as const;

    for (const r of reasons) {
      expect(translations.en.refusalReasons[r]).toBeTruthy();
      expect(translations.ar.refusalReasons[r]).toBeTruthy();
    }
  });

  it("has non-empty example queries for both languages", () => {
    expect(translations.en.examples.length).toBeGreaterThanOrEqual(4);
    expect(translations.ar.examples.length).toBeGreaterThanOrEqual(4);

    translations.en.examples.forEach((ex) => {
      expect(ex.query.length).toBeGreaterThan(0);
      expect(ex.badge).toBeTruthy();
    });
  });
});
