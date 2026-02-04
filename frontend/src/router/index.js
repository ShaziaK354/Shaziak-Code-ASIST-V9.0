import { createRouter, createWebHistory } from 'vue-router';
import PortfolioPage from '@/views/PortfolioPage.vue';
import DashboardPage from '@/views/DashboardPage.vue';

const routes = [
  {
    path: '/',
    name: 'Portfolio',
    component: PortfolioPage,
    meta: { title: 'Portfolio - ASIST' }
  },
  {
    path: '/dashboard/:caseId?',
    name: 'Dashboard',
    component: DashboardPage,
    meta: { title: 'Dashboard - ASIST' }
  },
  {
    path: '/case/:caseId',
    name: 'CaseView',
    component: DashboardPage,
    meta: { title: 'Case - ASIST' }
  },
  // Catch-all redirect to Portfolio
  {
    path: '/:pathMatch(.*)*',
    redirect: '/'
  }
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
});

// Update page title on navigation
router.beforeEach((to, from, next) => {
  document.title = to.meta.title || 'ASIST';
  next();
});

export default router;