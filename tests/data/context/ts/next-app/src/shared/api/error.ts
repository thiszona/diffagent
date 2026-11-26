/** An error the API returned, with its slug. */
export interface ApiError {
  slug: string;
  message: string;
}

/** Whether `e` is an {@link ApiError}. */
export function isApiError(e: unknown): e is ApiError {
  return typeof e === 'object' && e !== null && 'slug' in e;
}
