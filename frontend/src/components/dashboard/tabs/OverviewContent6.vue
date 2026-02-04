<template>
  <div class="tab-content-section-wrapper">
    <!-- Stats Cards Section (NEW) -->
    <div class="stats-cards-section">
      <div class="stats-card">
        <div class="stats-label">Total Countries</div>
        <div class="stats-value">4</div>
      </div>
      <div class="stats-card">
        <div class="stats-label">Total Cases</div>
        <div class="stats-value">4</div>
      </div>
      <div class="stats-card">
        <div class="stats-label">Total Portfolio Value</div>
        <div class="stats-value">$1.991B</div>
      </div>
      <div class="stats-card">
        <div class="stats-label">Total Obligations</div>
        <div class="stats-value">$1.546B</div>
      </div>
    </div>

    <!-- Alerts & Insights Section -->
    <div class="alerts-section">
      <div class="alerts-header">
        Alerts & Insights (sorted by due dates)
      </div>
      <div class="alerts-list">
        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>9/30/2025:</strong> N0002425WX12345 for NAVSEA Salary expired
            <a href="#" class="alert-link">(N-ERP)</a> - Consider pulling back additional funding
          </div>
        </div>

        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>11/23/2025:</strong> Action items 004, 005, 007, 011, 016, 026 responses are due
            <a href="#" class="alert-link">(View Report)</a>
          </div>
        </div>

        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>11/27/2025: <span class="yellow-highlight">YELLOW</span></strong> CEPT violation will appear on Lines 001 & 003 - CP-COM
            <a href="#" class="alert-link">(View report)</a>
          </div>
        </div>

        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>11/30/2025:</strong> ACR is due
            <a href="#" class="alert-link">(SCIP)</a>
          </div>
        </div>

        <div class="alert-item">
          <span class="alert-bullet">➤</span>
          <div class="alert-content">
            <strong>3/31/2026:</strong> Line 005 expires - Consider processing a modification to extend MOS
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';

// Props will be defined here later, e.g., to receive case data
// defineProps({ caseData: Object });
</script>

<style scoped>
.tab-content-section-wrapper {
  padding: 0;
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* ===== STATS CARDS SECTION ===== */
.stats-cards-section {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 8px;
}

.stats-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  transition: all 0.2s ease;
}

.stats-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.stats-label {
  font-size: 12px;
  font-weight: 600;
  color: #456379;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}

.stats-value {
  font-size: 28px;
  font-weight: 700;
  color: #1a1a1a;
  line-height: 1;
}

/* Optional: Color coding for different cards */
.stats-card:nth-child(1) .stats-value {
  color: #3498db;
}

.stats-card:nth-child(2) .stats-value {
  color: #2ecc71;
}

.stats-card:nth-child(3) .stats-value {
  color: #e74c3c;
}

.stats-card:nth-child(4) .stats-value {
  color: #f39c12;
}

/* Responsive stats cards */
@media (max-width: 1200px) {
  .stats-cards-section {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .stats-cards-section {
    grid-template-columns: 1fr;
  }
}

/* ===== ALERTS SECTION ===== */
.alerts-section {
  background: white;
  border: 2px solid #dc2626;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(220, 38, 38, 0.1);
  flex-grow: 1;
}

.alerts-header {
  background: white;
  padding: 12px 20px;
  font-weight: 700;
  color: #dc2626;
  font-size: 14px;
  border-bottom: 2px solid #dc2626;
}

.alerts-list {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: white;
}

.alert-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  line-height: 1.6;
}

.alert-bullet {
  color: #dc2626;
  font-size: 12px;
  flex-shrink: 0;
  margin-top: 2px;
}

.alert-content {
  color: #1a1a1a;
  font-size: 13px;
}

.alert-content strong {
  font-weight: 600;
  color: #000;
}

.alert-link {
  color: #0066cc;
  text-decoration: none;
  font-weight: 500;
}

.alert-link:hover {
  text-decoration: underline;
}

/* YELLOW highlight styling */
.yellow-highlight {
  background-color: #fef08a;
  color: #854d0e;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 700;
}
</style>
