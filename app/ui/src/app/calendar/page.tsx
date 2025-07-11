import CalendarView from './CalendarView';

export default function CalendarPage() {
  return (
    <div className="h-full flex flex-col p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Economic Calendar</h1>
      <div className="bg-[#23272F] rounded-lg p-6 flex-1">
        <CalendarView />
      </div>
    </div>
  );
} 