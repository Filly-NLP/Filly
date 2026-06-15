import { useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import WritePage from './pages/WritePage.jsx'
import AnalyticsPage from './pages/AnalyticsPage.jsx'
import GuidePage from './pages/GuidePage.jsx'
import './App.css'

export default function App() {
  const [activePage, setActivePage] = useState('write')

  const pages = {
    write: <WritePage />,
    analytics: <AnalyticsPage />,
    guide: <GuidePage />,
  }

  return (
    <div className="appShell">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <main className="appMain">
        {pages[activePage]}
      </main>
    </div>
  )
}
