import styles from './Sidebar.module.css'

const NAV_ITEMS = [
  {
    id: 'write',
    label: 'Write',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 20h9" />
        <path d="m16.5 3.5 4 4L8 20l-5 1 1-5 12.5-12.5Z" />
      </svg>
    ),
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 12a9 9 0 1 1-9-9v9h9Z" />
        <path d="M12 3a9 9 0 0 1 9 9" />
      </svg>
    ),
  },
  {
    id: 'guide',
    label: 'Guide',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15Z" />
      </svg>
    ),
  },
]

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <FillyLogo />
        <div>
          <div className={styles.logoName}>FILLY</div>
          <div className={styles.logoCatch}>Meow catchphrase</div>
        </div>
      </div>

      <hr className={styles.divider} />

      <nav className={styles.nav}>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`${styles.navItem} ${activePage === item.id ? styles.active : ''}`}
            onClick={() => onNavigate(item.id)}
            type="button"
          >
            <span className={styles.navIcon}>{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>
    </aside>
  )
}

function FillyLogo() {
  return (
    <svg className={styles.logoIcon} width="64" height="54" viewBox="0 0 96 80" aria-hidden="true">
      <path d="M12 29 48 8l36 21v36H12V29Z" fill="#f4e3c8" />
      <path d="M12 29h72L48 61 12 29Z" fill="#f7ead8" />
      <path d="M18 20h60v37H18V20Z" fill="#ffffff" />
      <path d="M18 20 48 45 78 20v37H18V20Z" fill="#f8f0e3" />
      <path d="M12 65 41 40l7 6 7-6 29 25H12Z" fill="#fff5df" />
      <path d="M18 20 48 45 78 20" fill="none" stroke="#e2d1b8" strokeWidth="2" />
    </svg>
  )
}
