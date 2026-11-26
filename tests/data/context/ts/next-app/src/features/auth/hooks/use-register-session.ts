'use client';

import { useEffect, useState } from 'react';
import { useRouter } from '@/i18n';
import { clearAuthSession, getRegisterSession } from '../session';

/** The registration OTP session, or a redirect to /register once it lapsed. */
export function useRegisterSession() {
  const router = useRouter();
  const [session, setSession] = useState<{ expired_at: number } | null>(null);

  useEffect(() => {
    const data = getRegisterSession();
    if (!data || Date.now() / 1000 > data.expired_at) {
      clearAuthSession();
      router.replace('/register');
      return;
    }
    setSession(data);
  }, []);

  return { session, setSession };
}
