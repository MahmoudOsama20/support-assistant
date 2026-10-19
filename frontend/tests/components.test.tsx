import { describe, it, expect } from "vitest";
import React from "react";
import { render, screen } from "@testing-library/react";
import { formatAsCsv } from "../components/results/csv-export";
import { SourceCard } from "../components/evidence/source-card";
import { SqlTable } from "../components/results/sql-table";
import { LanguageProvider } from "../lib/i18n/context";

describe("UI Components & Helpers", () => {
  it("formats CSV correctly with comma escaping and null values", () => {
    const columns = ["id", "name", "notes"];
    const rows = [
      [1, "Mona, Adel", "Active account"],
      [2, 'Omar "Fast" Fathy', null],
    ];

    const csv = formatAsCsv(columns, rows);
    const lines = csv.split("\n");

    expect(lines[0]).toBe("id,name,notes");
    expect(lines[1]).toBe('1,"Mona, Adel",Active account');
    expect(lines[2]).toBe('2,"Omar ""Fast"" Fathy",');
  });

  it("renders SourceCard with doc_id, title, and snippet", () => {
    const citation = {
      doc_id: "kb_001",
      chunk_id: "kb_001#c1",
      title: "Account Opening and Identity Verification",
      heading: "Verification levels",
      status: "active",
      snippet: "Nile Wallet has three account tiers: Basic, Plus, and Premium.",
    };

    render(
      <LanguageProvider>
        <SourceCard citation={citation} index={0} />
      </LanguageProvider>
    );

    expect(screen.getByText("Account Opening and Identity Verification")).toBeDefined();
    expect(screen.getByText("Verification levels")).toBeDefined();
    expect(screen.getByText("kb_001")).toBeDefined();
    expect(screen.getByText(/Nile Wallet has three account tiers/)).toBeDefined();
  });

  it("renders SqlTable with headers, rows, and count", () => {
    const columns = ["customer_id", "full_name", "tier", "balance"];
    const rows = [
      [1, "Mona Adel", "Premium", 84250.5],
      [2, "Omar Fathy", "Basic", 2300.0],
    ];

    render(
      <LanguageProvider>
        <SqlTable columns={columns} rows={rows} rowCount={2} />
      </LanguageProvider>
    );

    expect(screen.getByText("customer_id")).toBeDefined();
    expect(screen.getByText("Mona Adel")).toBeDefined();
    expect(screen.getByText("Omar Fathy")).toBeDefined();
    expect(screen.getByText(/84250.5/)).toBeDefined();
  });
});
