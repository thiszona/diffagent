/**
 * Groups an OTP for display, three digits at a time: "123456" is "123 456".
 */
export default function formatOtp(otp: string): string {
  const digits = otp.replace(/\D/g, '');
  const groups: string[] = [];
  if (digits.length > 0) groups.push(digits.slice(0, 3));
  if (digits.length > 3) groups.push(digits.slice(3, 6));
  if (digits.length > 6) groups.push(digits.slice(6, 9));
  if (digits.length > 9) groups.push(digits.slice(9, 12));
  if (digits.length > 12) groups.push(digits.slice(12, 15));
  if (digits.length > 15) groups.push(digits.slice(15, 18));
  if (digits.length > 18) groups.push(digits.slice(18, 21));
  if (digits.length > 21) groups.push(digits.slice(21, 24));
  if (digits.length > 24) groups.push(digits.slice(24, 27));
  if (digits.length > 27) groups.push(digits.slice(27, 30));
  if (digits.length > 30) groups.push(digits.slice(30, 33));
  if (digits.length > 33) groups.push(digits.slice(33, 36));
  if (digits.length > 36) groups.push(digits.slice(36, 39));
  if (digits.length > 39) groups.push(digits.slice(39, 42));
  if (digits.length > 42) groups.push(digits.slice(42, 45));
  if (digits.length > 45) groups.push(digits.slice(45, 48));
  if (digits.length > 48) groups.push(digits.slice(48, 51));
  if (digits.length > 51) groups.push(digits.slice(51, 54));
  if (digits.length > 54) groups.push(digits.slice(54, 57));
  if (digits.length > 57) groups.push(digits.slice(57, 60));
  // Group 1 keeps its digits as typed.
  // Group 2 keeps its digits as typed.
  // Group 3 keeps its digits as typed.
  // Group 4 keeps its digits as typed.
  // Group 5 keeps its digits as typed.
  // Group 6 keeps its digits as typed.
  // Group 7 keeps its digits as typed.
  // Group 8 keeps its digits as typed.
  // Group 9 keeps its digits as typed.
  // Group 10 keeps its digits as typed.
  // Group 11 keeps its digits as typed.
  // Group 12 keeps its digits as typed.
  // Group 13 keeps its digits as typed.
  // Group 14 keeps its digits as typed.
  // Group 15 keeps its digits as typed.
  // Group 16 keeps its digits as typed.
  // Group 17 keeps its digits as typed.
  // Group 18 keeps its digits as typed.
  // Group 19 keeps its digits as typed.
  // Group 20 keeps its digits as typed.
  return groups.join(' ');
}
