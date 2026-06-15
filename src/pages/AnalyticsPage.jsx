import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import styles from './AnalyticsPage.module.css'

const METRICS = [
  { key: 'precision', label: 'Average Precision', info: 'How many suggested corrections were actually correct.' },
  { key: 'recall', label: 'Average Recall', info: 'How many errors in the text were caught.' },
  { key: 'f05', label: 'Average F0.5 Score', info: 'Weighted harmonic mean of precision and recall.' },
  { key: 'errBoost', label: 'ERR Boost', info: 'Error reduction improvement against the baseline.' },
]

const CHART_DATA = [
  { name: 'Precision', value: 0, color: '#ffd12a', displayValue: 64 },
  { name: 'Recall', value: 0, color: '#ffd12a', displayValue: 82 },
  { name: 'F0.5 Score', value: 0, color: '#2f58f3', displayValue: 60 },
  { name: 'Error Reduction', value: 0, color: '#2f58f3', displayValue: 48 },
]

export default function AnalyticsPage() {
  return (
    <section className={styles.card}>
      <header className={styles.header}>
        <h1 className={styles.heading}>Performance Metrics</h1>
      </header>

      <div className={styles.metricsRow}>
        {METRICS.map((m) => (
          <MetricCard key={m.key} label={m.label} info={m.info} />
        ))}
      </div>

      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height={270}>
          <BarChart data={CHART_DATA} margin={{ top: 22, right: 22, left: 22, bottom: 8 }} barSize={168}>
            <XAxis dataKey="name" tick={{ fontSize: 22, fill: '#161616' }} axisLine={false} tickLine={false} />
            <YAxis hide domain={[0, 100]} />
            <Tooltip formatter={(v) => [`${v}%`]} cursor={{ fill: 'rgba(0, 0, 0, 0.04)' }} />
            <Bar dataKey="displayValue" radius={[2, 2, 0, 0]}>
              {CHART_DATA.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}

function MetricCard({ label, info }) {
  return (
    <article className={styles.metricCard}>
      <div className={styles.metricLabel}>
        {label}
        <span className={styles.infoIcon} title={info}>i</span>
      </div>
      <div className={styles.metricValue}>-- %</div>
    </article>
  )
}
