import { computeEarliestStart } from '@/lib/time';

export const start = (now: Date) => computeEarliestStart(now, 4);
