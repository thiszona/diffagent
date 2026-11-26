import { computeEarliestStart as earliest } from '@/lib/time';

export const scheduleStart = (now: Date) => earliest(now, 4);
