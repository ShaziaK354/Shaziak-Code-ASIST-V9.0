<template>
  <div class="portfolio-page" :style="{ marginLeft: sidebarMarginLeft }">
    <div class="portfolio-content-area">
      <PortfolioView 
        @case-clicked="handleCaseSelection" 
        :style="{ flexGrow: contentFlexGrow, transition: 'flex-grow 0.3s ease' }"
        class="content-panel"
        :class="{ 'expanded': isChatCollapsed }"
      />

      <div
        v-if="!isChatCollapsed"
        class="panel-resizer"
        @click="togglePanelExpansion"
        :title="isChatPanelExpanded ? 'Expand Portfolio Panel' : 'Expand Chat Panel'"
      >
        <i :class="isChatPanelExpanded ? 'fas fa-angle-right' : 'fas fa-angle-left'"></i>
      </div>

      <ChatInterface
        :style="{ flexGrow: chatFlexGrow, transition: 'flex-grow 0.3s ease' }"
        class="content-panel"
        :class="{ 'panel-collapsed': !isChatPanelExpanded && totalFlexGrow > 2, 'fully-collapsed': isChatCollapsed }"
        @panel-collapsed="handleChatPanelCollapsed"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import PortfolioView from '@/components/dashboard/tabs/PortfolioView.vue'
import ChatInterface from '@/components/chat/ChatInterface.vue'

const props = defineProps({
  isSidebarCollapsed: {
    type: Boolean,
    default: false
  }
})

const router = useRouter()

// Panel expansion state
const isChatPanelExpanded = ref(false)
const isChatCollapsed = ref(false)

const baseContentFlex = 1.5
const baseChatFlex = 1
const expandedFlex = 2.5
const collapsedFlex = 0.5

// Match the same margin calculation as MainDashboardLayout
const sidebarMarginLeft = computed(() => {
  const expandedSidebarWidth = "250px"
  const collapsedSidebarWidth = "70px"
  return props.isSidebarCollapsed ? collapsedSidebarWidth : expandedSidebarWidth
})

const contentFlexGrow = computed(() => {
  if (isChatCollapsed.value) {
    return 1
  }
  return isChatPanelExpanded.value ? collapsedFlex : baseContentFlex
})

const chatFlexGrow = computed(() => {
  if (isChatCollapsed.value) {
    return 0
  }
  return isChatPanelExpanded.value ? expandedFlex : baseChatFlex
})

const totalFlexGrow = computed(() => contentFlexGrow.value + chatFlexGrow.value)

const togglePanelExpansion = () => {
  isChatPanelExpanded.value = !isChatPanelExpanded.value
  console.log(`[PortfolioPage] Chat panel expanded: ${isChatPanelExpanded.value}`)
}

const handleChatPanelCollapsed = (collapsed) => {
  isChatCollapsed.value = collapsed
  console.log(`[PortfolioPage] Chat panel fully collapsed: ${collapsed}`)

  if (collapsed) {
    isChatPanelExpanded.value = false
  }
}

function handleCaseSelection(caseId) {
  console.log('[PortfolioPage] Navigating to case:', caseId)
  router.push({ name: 'Dashboard', params: { caseId } })
}
</script>

<style scoped>
.portfolio-page {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #f8fafc;
  min-height: 100vh;
  transition: margin-left 0.3s ease;
  padding: var(--space-md);
}

.portfolio-content-area {
  display: flex;
  flex-grow: 1;
  overflow: hidden;
  align-items: stretch;
}

.content-panel {
  min-width: 400px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
}

.content-panel.expanded {
  flex-grow: 1 !important;
  min-width: 100%;
}

.content-panel.fully-collapsed {
  flex-grow: 0 !important;
  min-width: 50px !important;
  max-width: 50px !important;
  overflow: visible !important;
}

.panel-resizer {
  flex-shrink: 0;
  width: 12px;
  background-color: var(--light);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border);
  border-right: 1px solid var(--border);
  z-index: 10;
  transition: background-color 0.2s ease;
}

.panel-resizer:hover {
  background-color: #e0e0e0;
}

.panel-resizer i {
  color: var(--secondary);
  font-size: 0.9rem;
}
</style>