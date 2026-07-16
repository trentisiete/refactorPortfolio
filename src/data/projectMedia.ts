const projectMedia = {
  'surrogate-models': [
    'app_forrester_ei_decision_synthesis_step5.png',
    'app_forrester_gp_evolution_no_noise.png',
    'app_real_best_gp_parity_by_target.png',
    'app_real_dataset_targets_by_diet.png',
    'app_real_observed_equal_weight_ranking.png',
    'evolution_incumbent_best_model_by_benchmark.png',
    'evolution_relative_mae_best_model_by_benchmark.png',
    'factor_effects_paired_for_tfg.png',
    'fig_01_lodo_generalization_vs_dummy.png',
    'fig_02_error_by_left_out_diet.png',
    'fig_04_kernel_family_mae.png',
    'fig_05_uncertainty_vs_error.png',
    'fig_06_prediction_intervals_by_diet.png',
    'fig_08_ei_candidate_landscape.png',
    'fig_ei_demo.png',
    'fig_gp_prior_posterior.png',
    'fig_kernel_matern52.png',
    'final_relative_improvement_mae_by_benchmark.png',
    'final_relative_incumbent_improvement_by_benchmark.png',
    'mae_vs_probabilistic_diagnostics_by_step.png',
  ],
  roadguard: [
    'bootstrap_f1_ci.png',
    'calibration_reliability.png',
    'confidence_shift_by_correctness.png',
    'example_1_China_MotorBike.jpg',
    'example_3_China_MotorBike.jpg',
    'example_4_United_States.jpg',
    'example_5_United_States.jpg',
    'per_class_ap_by_country.png',
    'policy_comparison.png',
    'risk_coverage_curve.png',
    'stress_robustness.png',
  ],
  'bgg-review-intelligence': [],
  'diffusion-models': [
    'bpd.jpeg',
    'classifier_architecture_image.png',
    'comparativa_samplers_vp_lineal.png',
    'evolucion_muestras_vp_lineal.png',
    'fid.jpeg',
    'final_conditional_comparison.png',
    'image_dcfe22.png',
    'imputacion_cifar_mnist_3etapas.png',
    'is_media.jpeg',
    'is_std.jpeg',
    'loss_average_barplot_all_sdes.png',
    'loss_curves_all_sdes.png',
    'loss_final_barplot_all_sdes.png',
    'loss_violin_plot_all_sdes.png',
    'mi_diagrama_gantt.jpeg',
    'mi_diagrama_gantt2.jpeg',
  ],
} as const;

export type ProjectMediaId = keyof typeof projectMedia;

const projectFolders: Record<ProjectMediaId, string> = {
  'surrogate-models': 'surrogate_models',
  roadguard: 'roadguard',
  'bgg-review-intelligence': 'bgg_review_intelligence',
  'diffusion-models': 'difussion_models',
};

export function createProjectGallery(
  id: ProjectMediaId,
  altPrefix: string,
  captions: Partial<Record<string, string>> = {},
) {
  return projectMedia[id].map((file, index) => ({
    src: `/assets/articles/${projectFolders[id]}/${file}`,
    alt: captions[file] ?? `${altPrefix} ${index + 1}`,
    ...(captions[file] ? { caption: captions[file] } : {}),
  }));
}
