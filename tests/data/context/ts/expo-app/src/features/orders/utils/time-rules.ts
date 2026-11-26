export function computeElapsedDuration(
	startDate: string,
	startTime: string,
	endDate: string,
	endTime: string,
): { hours: number; minutes: number } {
	const start = new Date(`${startDate}T${startTime}:00`);
	const end = new Date(`${endDate}T${endTime}:00`);
	const minutes = Math.max(0, (end.getTime() - start.getTime()) / 60000);
	return { hours: Math.floor(minutes / 60), minutes: minutes % 60 };
}
