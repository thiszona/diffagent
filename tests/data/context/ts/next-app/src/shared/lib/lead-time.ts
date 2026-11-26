import { computeEarliestStart } from './time';

export const LEAD_TIME_HOURS = 4;

/** The earliest start after the standard lead time. */
export function earliestLeadStart(now: Date) {
  const lead = LEAD_TIME_HOURS;
  return computeEarliestStart(now, lead);
}
