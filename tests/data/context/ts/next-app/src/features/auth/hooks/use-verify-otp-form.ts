'use client';

import { useState } from 'react';
import * as store from '../session';
import { isApiError } from '@/api';
import { useIsExpired } from '@/hooks/use-payment-countdown';
import { useRouter } from '@/i18n';
import formatOtp from './format-otp';
import { useRegisterSession } from './use-register-session';

const OTP_LENGTH = 6;

export function useVerifyOtpForm() {
  const router = useRouter();
  const { session, setSession } = useRegisterSession();
  const [otp, setOtp] = useState('');
  const limit = OTP_LENGTH;

  const isSessionExpired = useIsExpired(session?.expired_at);
  const canSubmit = otp.length >= limit && !isSessionExpired;

  function handleError(err: unknown) {
    if (!isApiError(err)) store.clearAuthSession();
    router.replace('/register');
  }

  return { otp: formatOtp(otp), setOtp, setSession, canSubmit, handleError };
}
