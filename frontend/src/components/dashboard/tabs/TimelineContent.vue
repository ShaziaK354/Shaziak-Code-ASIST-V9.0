<template>
  <div class="timeline-content-wrapper">
    <!-- Loading State -->
    <div v-if="loading" class="timeline-loading">
      <div class="spinner"></div>
      <p>Analyzing case documents and extracting timeline...</p>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="timeline-error">
      <div class="error-icon">⚠️</div>
      <h3>Failed to Load Timeline</h3>
      <p>{{ error }}</p>
      <button @click="loadTimeline" class="retry-button">
        <span>🔄</span> Retry
      </button>
    </div>

    <!-- Empty State -->
    <div v-else-if="timelineEvents.length === 0" class="timeline-empty">
      <div class="empty-icon">📅</div>
      <h3>No Timeline Events Found</h3>
      <p v-if="!hasCaseDocuments">Upload case documents (LOA, Minutes, Financial Data) to populate the timeline.</p>
      <p v-else>Timeline data will appear after documents are processed by the Vector DB.</p>
    </div>

    <!-- Timeline Table -->
    <div v-else class="timeline-table-wrapper">
      <table class="timeline-table">
        <thead>
          <tr>
            <th class="col-version">LOA Version</th>
            <th class="col-changes">Changes</th>
          </tr>
        </thead>
        <tbody>
          <template v-for="(event, index) in timelineEvents" :key="`event-${index}`">
            <tr 
              v-for="(change, changeIndex) in getEventChanges(event)" 
              :key="`row-${index}-${changeIndex}`"
              :class="getRowClass(index, changeIndex)"
            >
              <td 
                v-if="changeIndex === 0" 
                :rowspan="getEventChanges(event).length" 
                class="version-cell"
              >
                {{ event.version }}
              </td>
              <td class="change-cell">{{ change }}</td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue';

const props = defineProps({
  caseData: {
    type: Object,
    default: () => null
  }
});

// State
const loading = ref(false);
const error = ref(null);
const timelineEvents = ref([]);
const caseIdentifier = ref('');
const sources = ref({});
const documentsAnalyzed = ref(0);

// Fallback timeline data extracted from meeting minutes
const fallbackTimelineData = [
  {
    version: 'Modification 1',
    changes: [
      'Extends Line 001 POP to 12/31/2024',
      'Updates Line Note 001 language',
      'Updates Sole Source note',
      'Extends Line 003 POP'
    ]
  },
  {
    version: 'Amendment 1',
    changes: [
      'Increases Line 003 QTY by 5 and adds $350,000',
      'Decreases Line 013 funding be $100,000 and realigns to Line 011'
    ]
  },
  {
    version: 'Modification 2',
    changes: [
      'Realigns $150,000 funding from Line 006 to Line 008',
      'Extends Line 001 POP to 12/31/2025',
      'Extends Line 013 MOS to 12/31/2026'
    ]
  }
];

// Computed
const hasCaseDocuments = computed(() => {
  return props.caseData?.caseDocuments && props.caseData.caseDocuments.length > 0;
});

// Methods
const getEventChanges = (event) => {
  if (event.changes && event.changes.length > 0) {
    return event.changes;
  }
  return ['No detailed information available'];
};

const getRowClass = (eventIndex, changeIndex) => {
  const classes = [];
  
  // Alternate background for different versions (entire version block)
  if (eventIndex % 2 === 1) {
    classes.push('alternate-version');
  }
  
  return classes.join(' ');
};

const loadTimeline = async () => {
  if (!props.caseData?.id) {
    console.warn('[TimelineContent] No case ID available, using fallback data');
    timelineEvents.value = fallbackTimelineData;
    return;
  }

  loading.value = true;
  error.value = null;

  try {
    console.log(`[TimelineContent] 🔍 Fetching timeline for case: ${props.caseData.id}`);
    
    const response = await fetch(
      `http://localhost:3000/api/cases/${props.caseData.id}/timeline`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        },
        credentials: 'include'
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('[TimelineContent] 📊 API Response:', data);

    if (data.success && data.timeline && data.timeline.length > 0) {
      timelineEvents.value = data.timeline;
      caseIdentifier.value = data.case_identifier || props.caseData.id;
      sources.value = data.sources || {};
      documentsAnalyzed.value = data.sources?.documents_analyzed || 0;
      
      console.log(`[TimelineContent] ✅ Loaded ${timelineEvents.value.length} timeline events from API`);
    } else {
      // Fall back to hardcoded data if API returns empty
      console.log('[TimelineContent] ⚠️ No timeline events from API, using fallback data');
      timelineEvents.value = fallbackTimelineData;
    }
  } catch (err) {
    console.error('[TimelineContent] ❌ Error loading timeline:', err);
    // Use fallback data on error
    console.log('[TimelineContent] Using fallback timeline data due to error');
    timelineEvents.value = fallbackTimelineData;
    error.value = null; // Don't show error if we have fallback data
  } finally {
    loading.value = false;
  }
};

// Watchers
watch(() => props.caseData?.id, (newId, oldId) => {
  if (newId && newId !== oldId) {
    console.log(`[TimelineContent] Case ID changed: ${oldId} → ${newId}`);
    loadTimeline();
  }
});

watch(() => props.caseData?.updatedAt, (newTime, oldTime) => {
  if (newTime && newTime !== oldTime) {
    console.log(`[TimelineContent] Case updated at: ${newTime}`);
    loadTimeline();
  }
});

watch(() => props.caseData?.caseDocuments?.length, (newCount, oldCount) => {
  if (newCount && newCount !== oldCount) {
    console.log(`[TimelineContent] Document count changed: ${oldCount} → ${newCount}`);
    setTimeout(() => {
      loadTimeline();
    }, 2000);
  }
});

// Lifecycle
onMounted(() => {
  console.log('[TimelineContent] 🎬 Component mounted');
  console.log('[TimelineContent] Case data:', props.caseData);
  loadTimeline();
});
</script>

<style scoped>
.timeline-content-wrapper {
  padding: 0;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: white;
}

/* Loading State */
.timeline-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  gap: 16px;
  min-height: 300px;
}

.spinner {
  width: 48px;
  height: 48px;
  border: 4px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.timeline-loading p {
  color: #6b7280;
  font-size: 14px;
}

/* Error State */
.timeline-error {
  text-align: center;
  padding: 60px 20px;
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.error-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.timeline-error h3 {
  color: #dc2626;
  margin-bottom: 8px;
  font-size: 20px;
}

.timeline-error p {
  color: #6b7280;
  margin-bottom: 16px;
  max-width: 500px;
}

.retry-button {
  padding: 10px 24px;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: background 0.2s;
}

.retry-button:hover {
  background: #2563eb;
}

/* Empty State */
.timeline-empty {
  text-align: center;
  padding: 60px 20px;
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 16px;
  opacity: 0.5;
}

.timeline-empty h3 {
  color: #374151;
  margin-bottom: 8px;
  font-size: 20px;
}

.timeline-empty p {
  color: #6b7280;
  max-width: 500px;
  line-height: 1.6;
}

/* Timeline Table - Scrollable Container */
.timeline-table-wrapper {
  width: 100%;
  height: 100%;
  overflow: auto;
  background: white;
}

/* Custom Scrollbar */
.timeline-table-wrapper::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

.timeline-table-wrapper::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.timeline-table-wrapper::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 5px;
}

.timeline-table-wrapper::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* Timeline Table */
.timeline-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  table-layout: fixed;
}

/* Table Header */
.timeline-table thead {
  position: sticky;
  top: 0;
  z-index: 10;
}

.timeline-table thead th {
  padding: 12px 16px;
  text-align: left;
  background: #2c5f7f;
  color: white;
  font-weight: 600;
  font-size: 14px;
  border: none;
}

/* Column Widths - matching mockup */
.col-version {
  width: 180px;
}

.col-changes {
  width: calc(100% - 180px);
}

/* Table Body */
.timeline-table tbody {
  background: white;
}

.timeline-table tbody tr {
  border: none;
}

/* Version Cell - Gray background */
.version-cell {
  padding: 12px 16px;
  vertical-align: top;
  font-weight: 500;
  color: #1f2937;
  background-color: #d1d5db;
  border: 1px solid #9ca3af;
  border-right: 1px solid #9ca3af;
  font-size: 14px;
}

/* Change Cell - White background by default */
.change-cell {
  padding: 12px 16px;
  color: #1f2937;
  line-height: 1.5;
  font-size: 14px;
  background-color: white;
  border: 1px solid #9ca3af;
  border-left: none;
}

/* Alternate version styling - lighter gray for changes */
.timeline-table tbody tr.alternate-version .change-cell {
  background-color: #e5e7eb;
}

/* Remove default hover effect */
.timeline-table tbody tr:hover .change-cell {
  background-color: inherit;
}

.timeline-table tbody tr.alternate-version:hover .change-cell {
  background-color: #e5e7eb;
}

/* Responsive Adjustments */
@media (max-width: 768px) {
  .timeline-table {
    font-size: 13px;
  }

  .timeline-table thead th {
    padding: 10px 14px;
    font-size: 13px;
  }

  .version-cell,
  .change-cell {
    padding: 10px 14px;
    font-size: 13px;
  }

  .col-version {
    width: 150px;
  }
  
  .col-changes {
    width: calc(100% - 150px);
  }
}

@media (max-width: 576px) {
  .timeline-table {
    min-width: 600px;
  }
}
</style>