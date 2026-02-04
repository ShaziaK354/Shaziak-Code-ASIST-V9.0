<template>
  <section class="case-details-panel" v-if="displayCaseData">
    <div class="case-header">
      <div class="case-title-main-area"> 
        <div class="case-title-container">
          <h2 id="activeCaseTitle">
            <span id="activeCaseNumberDisplay" class="case-number-title">{{ displayCaseData.caseNumber || 'N/A' }}:</span>
            <span id="activeCaseDescriptionDisplay" class="case-description-title">{{ displayCaseData.caseDescription || 'No Description' }}</span>
          </h2>
          <div class="search-bar-case">
            <span class="nav-icon"><i class="fas fa-search"></i></span>
            <input type="text" placeholder="Search this case..." title="Search within this case">
          </div>
        </div>
        <div class="case-meta">
          <div class="case-meta-item" title="Creation Date">
            <span class="nav-icon"><i class="far fa-calendar-alt"></i></span>
            <span>Created: {{ formatDate(displayCaseData.createdAt) }}</span>
          </div>
          <div class="case-meta-item" title="Case Value">
            <span class="nav-icon"><i class="fas fa-dollar-sign"></i></span>
            <span>Value: {{ formatCurrency(displayCaseData.estimatedValue) }}</span>
          </div>
          <div class="case-meta-item" title="LOA Status">
            <span class="nav-icon"><i class="far fa-file-alt"></i></span>
            <span>LOA: {{ displayCaseData.loaStatus || 'N/A' }}</span>
          </div>
        </div>
      </div>
      <div class="case-status">
        <div class="status-badge" :class="statusBadgeClass">{{ displayCaseData.status || 'Unknown' }}</div>
      </div>
    </div>

    <nav class="tabs">
      <div 
        v-for="tab in tabs" 
        :key="tab.id" 
        class="tab" 
        :class="{ active: localActiveTab === tab.id }"
        @click="selectTab(tab.id)"
        role="tab"
        :aria-selected="localActiveTab === tab.id"
        :aria-controls="`${tab.id}ContentVue`" 
      >
        {{ tab.name }}
      </div>
    </nav>
  </section>
  <section v-else class="case-details-panel placeholder">
    <p>No case selected or case data not available.</p>
    <p v-if="caseStore.isLoadingCases">Loading cases...</p>
    <p v-if="caseStore.caseError">Error loading case data: {{ caseStore.caseError }}</p>
  </section>
</template>

<script setup>
import { ref, defineProps, defineEmits, computed, watch, onMounted } from 'vue';
import { useCaseStore } from '@/stores/caseStore';

const props = defineProps({
  activeCaseData: {
    type: Object,
    default: () => null
  }
});

const emit = defineEmits(['tab-selected']);
const caseStore = useCaseStore();

// ===================================================================
// ✅ DATA OVERRIDE: SR-P-NAV LOA Data (from uploaded PDF)
// ===================================================================
const displayCaseData = computed(() => {
  if (!props.activeCaseData) return null;
  
  // Override with SR-P-NAV data from LOA
  return {
    ...props.activeCaseData,
    id: 'sr-p-nav-001',
    caseNumber: 'SR-P-NAV',
    caseDescription: 'Naval Strike Missiles (NSM), associated support equipment, training, and technical services',
    createdAt: '2025-03-18T00:00:00Z', // Based on SR/RSNF/4721 dated 18 March 2025
    estimatedValue: 284961260, // Total Estimated Cost: $284,961,260
    loaStatus: 'Implemented', // Implementation Date: 31 May 2025
    implementationDate: '2025-05-31T00:00:00Z',
    offerExpirationDate: '2025-06-15T00:00:00Z',
    initialDeposit: 12450870, // Initial Deposit: $12,450,870
    congressionalNotification: '25-42',
    status: 'Active',
    purchaser: 'Embassy of Saudi Arabia, Office of the Naval Attaché',
    implementingAgency: 'Navy International Programs Office (NIPO)',
    termsOfSale: 'Cash Prior to Delivery, Dependable Undertaking',
    lineItems: [
      {
        itemNumber: '001',
        description: 'NAVAL STRIKE MISSILE, TACTICAL',
        partNumber: 'G2H 9G2H00NSMTACT',
        quantity: 96,
        unitOfIssue: 'EA',
        unitCost: 1875000,
        totalCost: 180000000,
        sourceCode: 'X',
        mos: '24-36',
        ta: 'TA4',
        orc: 'Z',
        dtc: '8'
      },
      {
        itemNumber: '002',
        description: 'NAVAL STRIKE MISSILE CONTAINERS',
        partNumber: 'G2H 810000NSMCONT',
        quantity: 12,
        unitOfIssue: 'EA',
        unitCost: 125000,
        totalCost: 1500000,
        sourceCode: 'P',
        mos: '18-30',
        ta: 'TA5',
        orc: 'A',
        dtc: '2'
      },
      {
        itemNumber: '003',
        description: 'NAVAL STRIKE MISSILE GROUND SUPPORT EQUIPMENT',
        partNumber: 'G2H 9G2H00NSMGSE',
        quantity: null,
        unitOfIssue: 'XX',
        unitCost: null,
        totalCost: 8500000,
        sourceCode: 'P',
        mos: '24-36',
        ta: 'TA5',
        orc: 'A',
        dtc: '4'
      },
      {
        itemNumber: '004',
        description: 'TECHNICAL DOCUMENTATION',
        partNumber: 'R9Z 0H1F00TECHDOC',
        quantity: null,
        unitOfIssue: 'XX',
        unitCost: null,
        totalCost: 3200000,
        sourceCode: 'P',
        mos: '12-18',
        ta: 'TA5',
        orc: 'A',
        dtc: '4'
      },
      {
        itemNumber: '005',
        description: 'OFFICER TRAINING - NSM Tactical Employment Course',
        partNumber: 'R9Z 0L9T00TRNOFFC',
        quantity: 16,
        unitOfIssue: 'EA',
        unitCost: 45000,
        totalCost: 720000,
        sourceCode: 'S',
        mos: 'Sep 2025 - Mar 2026',
        ta: 'TA3',
        orc: 'A',
        dtc: '4'
      },
      {
        itemNumber: '006',
        description: 'ENLISTED TRAINING - NSM Maintenance Course',
        partNumber: 'R9Z 0L9T00TRNENL',
        quantity: 24,
        unitOfIssue: 'EA',
        unitCost: 28000,
        totalCost: 672000,
        sourceCode: 'S',
        mos: 'Oct 2025 - Apr 2026',
        ta: 'TA3',
        orc: 'A',
        dtc: '4'
      },
      {
        itemNumber: '007',
        description: 'TECHNICAL SERVICES - NSM Integration Support',
        partNumber: 'R9Z 079Z00TECHSVC',
        quantity: null,
        unitOfIssue: 'XX',
        unitCost: null,
        totalCost: 42500000,
        sourceCode: 'S',
        mos: 'May 2025 - Dec 2027',
        ta: 'TA3',
        orc: 'A',
        dtc: '4'
      },
      {
        itemNumber: '008',
        description: 'LOGISTICS SUPPORT SERVICES',
        partNumber: 'R9Z 079Z00LOGSUPP',
        quantity: null,
        unitOfIssue: 'XX',
        unitCost: null,
        totalCost: 18400000,
        sourceCode: 'S',
        mos: 'Jun 2025 - May 2028',
        ta: 'TA3',
        orc: 'A',
        dtc: '4'
      }
    ],
    costSummary: {
      netEstimatedCost: 255492000,
      packingCratingHandling: 6687220,
      administrativeCharge: 8175744,
      transportation: 14606296,
      other: 0,
      totalEstimatedCost: 284961260
    },
    paymentSchedule: [
      { date: 'Initial Deposit', quarterly: 12450870, cumulative: 12450870 },
      { date: '15 Sep 2025', quarterly: 8547265, cumulative: 20998135 },
      { date: '15 Dec 2025', quarterly: 14285420, cumulative: 35283555 },
      { date: '15 Mar 2026', quarterly: 22145680, cumulative: 57429235 },
      { date: '15 Jun 2026', quarterly: 38642150, cumulative: 96071385 },
      { date: '15 Sep 2026', quarterly: 45287920, cumulative: 141359305 },
      { date: '15 Dec 2026', quarterly: 52148735, cumulative: 193508040 },
      { date: '15 Mar 2027', quarterly: 48625410, cumulative: 242133450 },
      { date: '15 Jun 2027', quarterly: 42827810, cumulative: 284961260 }
    ]
  };
});
// ===================================================================

const tabs = ref([
  { id: 'overview', name: 'Overview' },
  { id: 'timeline', name: 'Timeline' },
  { id: 'financial', name: 'Financial' },
  { id: 'logistics', name: 'Logistics' },
  { id: 'reports', name: 'Reports' },
  { id: 'documents', name: 'Documents' }
]);

const localActiveTab = ref('overview'); 

const selectTab = (tabId) => {
  localActiveTab.value = tabId;
  emit('tab-selected', tabId); 
};

const statusBadgeClass = computed(() => {
  if (!displayCaseData.value || !displayCaseData.value.status) return 'status-unknown';
  const status = displayCaseData.value.status.toLowerCase();
  if (status === 'active' || status === 'imported') return 'status-active';
  if (status === 'implemented') return 'status-implemented';
  if (status === 'closed') return 'status-closed';
  return 'status-other';
});

const formatCurrency = (value) => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return 'N/A';
  }
  const numberValue = Number(value);
  return `$${numberValue.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
};

const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit'
        });
    } catch (e) {
        return 'Invalid Date';
    }
};

watch(() => props.activeCaseData, (newCaseData, oldCaseData) => {
  console.log("[CaseDetailsPanel] activeCaseData prop updated:", newCaseData);
  if (newCaseData && (!oldCaseData || newCaseData.id !== oldCaseData.id || newCaseData.caseNumber !== oldCaseData.caseNumber)) {
    localActiveTab.value = 'overview'; 
    emit('tab-selected', 'overview'); 
  } else if (newCaseData && !oldCaseData) { 
    emit('tab-selected', localActiveTab.value);
  }
}, { immediate: true, deep: true });

onMounted(() => {
    if (displayCaseData.value) {
        console.log("[CaseDetailsPanel] Mounted with case:", displayCaseData.value.caseNumber);
    } else {
        console.log("[CaseDetailsPanel] Mounted with no active case data.");
    }
});

</script>

<style scoped>
.case-details-panel {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  padding: var(--space-md);
  margin-bottom: var(--space-sm); 
  flex-shrink: 0; 
}
.case-details-panel.placeholder {
    text-align: center;
    color: #777;
    padding: var(--space-lg);
}

.case-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start; 
  margin-bottom: var(--space-sm);
  gap: var(--space-md);
}

.case-title-main-area {
    flex-grow: 1;
    min-width: 0;
}

.case-title-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-grow: 1;
  gap: var(--space-md);
  margin-bottom: var(--space-xs);
}

.case-title-container h2 {
  font-size: 1.3rem;
  display: flex;
  align-items: center;
  margin: 0;
  flex-shrink: 1;
  min-width: 0;
}
.case-number-title {
    font-weight: 600;
    margin-right: var(--space-xs);
    white-space: nowrap;
}
.case-description-title {
    font-weight: normal;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-dark);
}

.search-bar-case {
  display: flex;
  align-items: center;
  background-color: #f0f0f0;
  border-radius: 4px;
  padding: var(--space-xs) var(--space-sm);
  width: 250px;
  max-width: 300px;
  border: 1px solid var(--border);
  flex-shrink: 0;
}
.search-bar-case .nav-icon { 
  color: var(--text-dark);
  margin-right: var(--space-xs);
  font-size: 0.9rem;
}
.search-bar-case input {
  border: none;
  outline: none;
  flex: 1;
  font-size: 0.85rem;
  background: transparent;
}

.case-meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
  font-size: 0.85rem;
  color: #666;
}
.case-meta-item {
  display: flex;
  align-items: center;
}
.case-meta-item .nav-icon { 
  margin-right: var(--space-xs);
  color: var(--accent);
  font-size: 0.9rem;
}

.case-status {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}
.status-badge {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  color: white;
}
.status-badge.status-active { background-color: var(--success); }
.status-badge.status-implemented { background-color: var(--accent); } 
.status-badge.status-closed { background-color: var(--secondary); } 
.status-badge.status-unknown, .status-badge.status-other { background-color: var(--border); color: var(--text-dark); }

.tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0; 
}
.tab {
  padding: var(--space-sm) var(--space-md);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  font-weight: 500;
  font-size: 0.9rem;
  color: var(--text-dark);
  transition: color 0.2s ease, border-bottom-color 0.2s ease;
}
.tab:hover {
  color: var(--accent);
}
.tab.active {
  border-bottom-color: var(--accent);
  color: var(--accent);
}
</style>