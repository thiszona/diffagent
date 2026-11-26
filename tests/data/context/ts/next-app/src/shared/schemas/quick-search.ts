import { computeEarliestStart } from '../lib';

export const quickSearch = {
  earliest: (now: Date) => computeEarliestStart(now, 4),
};
