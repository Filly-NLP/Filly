import styles from './GuidePage.module.css'

const STEPS = [
  {
    number: 1,
    title: 'Enter Your Text',
    icon: 'write',
    body: [
      'Type or paste your Filipino text into the text editor.',
      'Filly will automatically analyze your input and check for informal words, phrases, and grammatical issues.',
    ],
  },
  {
    number: 2,
    title: 'Review Suggestions',
    icon: 'book',
    body: [
      'Words or phrases that may need correction will be highlighted.',
      'Click on a highlighted suggestion to view the recommended correction.',
    ],
    bullets: ['"Accept" the suggested correction', '"Ignore" the suggestion if you prefer the original text'],
  },
  {
    number: 3,
    title: 'View Corrected Output',
    icon: 'check',
    body: [
      'Accepted corrections are automatically applied to your text.',
      'The updated version helps improve the formality and grammatical quality of your writing.',
    ],
  },
]

export default function GuidePage() {
  return (
    <article className={styles.card}>
      <div className={styles.content}>
        <h1 className={styles.heroTitle}>Welcome to Filly!</h1>
        <p className={styles.heroSub}>Your Filipino Writing Assistant</p>

        <section className={styles.section}>
          <h2 className={styles.sectionHeading}>What is Filly?</h2>
          <p className={styles.body}>
            Filly helps users improve their Filipino writing by detecting informal, colloquial, and
            grammatically incorrect words or phrases and suggesting more formal and appropriate alternatives.
          </p>
          <p className={styles.body}>
            Whether you're preparing academic papers, reports, official documents, or professional
            communications, Filly assists in making your Filipino text clearer, more formal, and
            grammatically correct.
          </p>
        </section>

        <section className={styles.section}>
          <h2 className={styles.sectionHeading}>How to Use Filly</h2>
          <div className={styles.steps}>
            {STEPS.map((step) => (
              <div key={step.number} className={styles.step}>
                <div className={styles.stepIconWrap}>
                  <StepIcon icon={step.icon} />
                </div>
                <div className={styles.stepContent}>
                  <h3 className={styles.stepTitle}>{step.number}. {step.title}</h3>
                  {step.body.map((p) => (
                    <p key={p} className={styles.body}>{p}</p>
                  ))}
                  {step.bullets && (
                    <div className={styles.bullets}>
                      <p>You may:</p>
                      <ul>
                        {step.bullets.map((b) => <li key={b}>{b}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        <aside className={styles.tipsBox}>
          <h3 className={styles.tipsTitle}>Tips for Best Results</h3>
          <p className={styles.body}>
            Filly provides intelligent suggestions based on its language models and correction rules.
            While the system aims to improve writing quality, users are encouraged to review all suggested
            corrections to ensure they match the intended meaning and context.
          </p>
        </aside>
      </div>
    </article>
  )
}

function StepIcon({ icon }) {
  const paths = {
    write: (
      <>
        <path d="M12 20h9" />
        <path d="m16.5 3.5 4 4L8 20l-5 1 1-5 12.5-12.5Z" />
      </>
    ),
    book: (
      <>
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15Z" />
      </>
    ),
    check: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="m8 12 3 3 5-6" />
      </>
    ),
  }

  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[icon]}
    </svg>
  )
}
