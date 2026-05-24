export type OfferFlowState = {
  rawText: string;
  source: string;
  amount?: number;
  termMonths?: number;
  earlyNeed: "yes" | "no" | "unknown";
  includeTaxes: boolean;
  includeInflation: boolean;
  parsedFields: Record<string, string | number | null>;
  unknownFields: Array<Record<string, unknown>>;
  assumptions: string[];
  userOverrides: Record<string, unknown>;
  calculationResult: Record<string, unknown>;
  reportPayload: Record<string, unknown>;
};

export const initialOfferFlowState: OfferFlowState = {
  rawText: "",
  source: "другое",
  earlyNeed: "unknown",
  includeTaxes: true,
  includeInflation: true,
  parsedFields: {},
  unknownFields: [],
  assumptions: [],
  userOverrides: {},
  calculationResult: {},
  reportPayload: {},
};
