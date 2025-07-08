export default function SettingsPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Settings</h1>
      <div className="bg-[#23272F] rounded-lg p-6">
        <p className="text-gray-300 text-lg">
          This is the settings page. Here you will find system configuration and user preferences.
        </p>
        <div className="mt-4 p-4 bg-[#181A20] rounded border border-gray-700">
          <p className="text-gray-400">
            Account settings, API configuration, notification preferences, and system options will be available here.
          </p>
        </div>
      </div>
    </div>
  );
} 