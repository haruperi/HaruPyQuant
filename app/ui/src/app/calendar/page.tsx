export default function CalendarPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Calendar</h1>
      <div className="bg-[#23272F] rounded-lg p-6">
        <p className="text-gray-300 text-lg">
          This is the calendar page. Here you will find economic calendar and trading events.
        </p>
        <div className="mt-4 p-4 bg-[#181A20] rounded border border-gray-700">
          <p className="text-gray-400">
            Economic calendar with high-impact news events and market holidays will be displayed here.
          </p>
        </div>
      </div>
    </div>
  );
} 