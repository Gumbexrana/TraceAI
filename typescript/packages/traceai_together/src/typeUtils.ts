/**
 * Type guard to check if a value is a string
 */
export function isString(value: unknown): value is string {
  return typeof value === "string";
