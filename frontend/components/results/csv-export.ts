/**
 * Safe CSV serialization of table columns and rows
 */
export function formatAsCsv(columns: string[], rows: (string | number | boolean | null)[][]): string {
  const escapeCell = (val: unknown): string => {
    if (val === null || val === undefined) return "";
    const str = String(val);
    if (str.includes(",") || str.includes('"') || str.includes("\n") || str.includes("\r")) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const headerLine = columns.map(escapeCell).join(",");
  const rowLines = rows.map((row) => row.map(escapeCell).join(","));

  return [headerLine, ...rowLines].join("\n");
}
