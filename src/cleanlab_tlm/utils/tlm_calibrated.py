"""
TLM Calibrated is a variant of the Trustworthy Language Model (TLM) that facilitates the calibration of trustworthiness scores
using existing ratings for prompt-response pairs, which allows for better alignment of the TLM scores in specialized-use cases.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from cleanlab_tlm.errors import (
    MissingApiKeyError,
    TlmNotCalibratedError,
    ValidationError,
)
from cleanlab_tlm.tlm import TLM, TLMOptions, TLMResponse, TLMScore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cleanlab_tlm.internal.types import TLMQualityPreset


class TLMCalibrated:
    def __init__(
        self,
        quality_preset: TLMQualityPreset = "medium",
        *,
        api_key: Optional[str] = None,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """
        lazydocs: ignore
        """
        try:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
        except ImportError:
            raise ImportError(
                "Cannot import scikit-learn which is required to use TLMCalibrated. "
                "Please install it using `pip install scikit-learn` and try again."
            )

        self._api_key = api_key or os.environ.get("CLEANLAB_TLM_API_KEY")
        if self._api_key is None:
            raise MissingApiKeyError

        if quality_preset not in {"base", "low", "medium"}:
            raise ValidationError(
                f"Invalid quality preset: {quality_preset}. TLMCalibrated only supports 'base', 'low' and 'medium' presets."
            )
        self._quality_preset = quality_preset

        self._options = options
        self._timeout = timeout if timeout is not None and timeout > 0 else None
        self._verbose = verbose

        custom_eval_criteria_list = self._options.get("custom_eval_criteria", []) if self._options else []

        # number of custom eval critera + 1 to account for the default TLM trustworthiness score
        self._num_features = len(custom_eval_criteria_list) + 1
        self._rf_model = RandomForestRegressor(monotonic_cst=[1] * self._num_features)

        self._tlm = TLM(
            quality_preset=self._quality_preset,
            api_key=self._api_key,
            options=self._options,
            timeout=self._timeout,
            verbose=self._verbose,
        )

    def fit(self, tlm_scores: list[TLMScore], ratings: Sequence[float]) -> None:
        """
        Callibrate the model using TLM scores obtained from a previous `TLM.get_trustworthiness_score()` call
        using the provided numeric ratings.

        Args:
            tlm_scores (list[TLMScore]): list of [TLMScore](../tlm/#class-tlmscore) object obtained
                from a previous `TLM.get_trustworthiness_score()` call
            ratings (Sequence[float]): sequence of numeric ratings corresponding to each prompt-response pair,
                the length of this sequence must match the length of the `tlm_scores`.
        """
        if len(tlm_scores) != len(ratings):
            raise ValidationError("The list of ratings must be of the same length as the list of TLM scores.")

        tlm_scores_df = pd.DataFrame(tlm_scores)
        extracted_scores = self._extract_tlm_scores(tlm_scores_df)

        if extracted_scores.shape[1] != self._num_features:
            raise ValidationError(
                f"TLMCalibrated has {self._num_features - 1} custom evaluation criteria defined, "
                f"however the tlm_scores provided have {extracted_scores.shape[1] - 1} custom evaluation scores. "
                "Please make sure the number of custom evaluation criterias match."
            )

        # using pandas so that NaN values are handled correctly
        ratings_series = pd.Series(ratings)
        ratings_normalized = (ratings_series - ratings_series.min()) / (ratings_series.max() - ratings_series.min())

        self._rf_model.fit(extracted_scores, ratings_normalized.values)

    def prompt(
        self, prompt: Union[str, Sequence[str]]
    ) -> Union[TLMResponseWithCalibration, list[TLMResponseWithCalibration]]:
        """
        Gets response and a calibrated trustworthiness score for the given prompts,
        make sure that the model has been calibrated by calling the `.fit()` method before using this method.

        Similar to [`TLM.prompt()`](../tlm/#method-prompt),
        view documentation there for expected input arguments and outputs.
        """
        try:
            from sklearn.exceptions import NotFittedError  # type: ignore
            from sklearn.utils.validation import check_is_fitted  # type: ignore
        except ImportError:
            raise ImportError(
                "Cannot import scikit-learn which is required to use TLMCalibrated. "
                "Please install it using `pip install scikit-learn` and try again."
            )

        try:
            check_is_fitted(self._rf_model)
        except NotFittedError:
            raise TlmNotCalibratedError(
                "TLMCalibrated has to be calibrated before prompting new data, use the .fit() method to calibrate the model."
            )
        tlm_response = self._tlm.prompt(prompt)

        is_single_query = isinstance(tlm_response, dict)
        if is_single_query:
            assert not isinstance(tlm_response, list)
            tlm_response = [tlm_response]
        tlm_response_df = pd.DataFrame(tlm_response)

        extracted_scores = self._extract_tlm_scores(tlm_response_df)

        tlm_response_df["calibrated_score"] = self._rf_model.predict(extracted_scores)

        if is_single_query:
            return cast(TLMResponseWithCalibration, tlm_response_df.to_dict(orient="records")[0])

        return cast(list[TLMResponseWithCalibration], tlm_response_df.to_dict(orient="records"))

    def get_trustworthiness_score(
        self, prompt: Union[str, Sequence[str]], response: Union[str, Sequence[str]]
    ) -> Union[TLMScoreWithCalibration, list[TLMScoreWithCalibration]]:
        """
        Computes the calibrated trustworthiness score for arbitrary given prompt-response pairs,
        make sure that the model has been calibrated by calling the `.fit()` method before using this method.

        Similar to [`TLM.get_trustworthiness_score()`](../tlm/#method-get_trustworthiness_score),
        view documentation there for expected input arguments and outputs.
        """
        try:
            from sklearn.exceptions import NotFittedError
            from sklearn.utils.validation import check_is_fitted
        except ImportError:
            raise ImportError(
                "Cannot import scikit-learn which is required to use TLMCalibrated. "
                "Please install it using `pip install scikit-learn` and try again."
            )

        try:
            check_is_fitted(self._rf_model)
        except NotFittedError:
            raise TlmNotCalibratedError(
                "TLMCalibrated has to be calibrated before scoring new data, use the .fit() method to calibrate the model."
            )

        tlm_scores = self._tlm.get_trustworthiness_score(prompt, response)

        is_single_query = isinstance(tlm_scores, dict)
        if is_single_query:
            assert not isinstance(tlm_scores, list)
            tlm_scores = [tlm_scores]
        tlm_scores_df = pd.DataFrame(tlm_scores)

        extracted_scores = self._extract_tlm_scores(tlm_scores_df)

        tlm_scores_df["calibrated_score"] = self._rf_model.predict(extracted_scores)

        if is_single_query:
            return cast(TLMScoreWithCalibration, tlm_scores_df.to_dict(orient="records")[0])

        return cast(list[TLMScoreWithCalibration], tlm_scores_df.to_dict(orient="records"))

    def _extract_tlm_scores(self, tlm_scores_df: pd.DataFrame) -> npt.NDArray[np.float64]:
        """
        Transform a DataFrame containing TLMScore objects into a 2D numpy array,
        where each column represents different scores including trustworthiness score and any custom evaluation criteria.

        Args:
            tlm_scores_df: DataFrame constructed using a list of TLMScore objects.

        Returns:
            np.ndarray: 2D numpy array where each column corresponds to different scores.
              The first column is the trustworthiness score, followed by any custom evaluation scores if present.
        """
        tlm_log = tlm_scores_df.get("log", None)

        # if custom_eval_criteria is present in the log, use it as features
        if tlm_log is not None and "custom_eval_criteria" in tlm_log.iloc[0]:
            custom_eval_scores = np.array(
                tlm_scores_df["log"]
                .apply(lambda x: [criteria["score"] for criteria in x["custom_eval_criteria"]])
                .tolist()
            )
            all_scores = np.hstack(
                [
                    tlm_scores_df["trustworthiness_score"].to_numpy().reshape(-1, 1),
                    custom_eval_scores,
                ]
            )
        # otherwise use the TLM trustworthiness score as the only feature
        else:
            all_scores = tlm_scores_df["trustworthiness_score"].to_numpy().reshape(-1, 1)

        return all_scores


class TLMResponseWithCalibration(TLMResponse):
    """
    A typed dict similar to [TLMResponse](../tlm/#class-tlmresponse) but containing an extra key `calibrated_score`.
    View [TLMResponse](../tlm/#class-tlmresponse) for the description of the other keys in this dict.

    Attributes:
        calibrated_score (float, optional): score between 0 and 1 that has been calibrated to the provided ratings.
        A higher score indicates a higher confidence that the response is correct/trustworthy.
    """

    calibrated_score: Optional[float]


class TLMScoreWithCalibration(TLMScore):
    """
    A typed dict similar to [TLMScore](../tlm/#class-tlmscore) but containing an extra key `calibrated_score`.
    View [TLMScore](../tlm/#class-tlmscore) for the description of the other keys in this dict.

    Attributes:
        calibrated_score (float, optional): score between 0 and 1 that has been calibrated to the provided ratings.
        A higher score indicates a higher confidence that the response is correct/trustworthy.
    """

    calibrated_score: Optional[float]


def _get_skops() -> Any:
    """Lazy import for skops to avoid unnecessary dependency."""
    try:
        import skops.io  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "The skops package is required for model serialization. Please install it with: pip install skops"
        )

    return skops.io


def save_tlm_calibrated_state(model: TLMCalibrated, filename: str) -> None:
    """Save fitted TLMCalibrated model state to file.

    Args:
        model (TLMCalibrated): A fitted TLMCalibrated model instance
        filename (str): Path where the model state will be saved

    Raises:
        sklearn.exceptions.NotFittedError: If the model has not been fitted
        ImportError: If skops or sklearn package is not installed
    """
    try:
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted
    except ImportError:
        raise ImportError(
            "Cannot import scikit-learn which is required to use TLMCalibrated. "
            "Please install it using `pip install scikit-learn` and try again."
        )

    # Verify model is fitted
    rf_model = model._rf_model  # noqa: SLF001
    try:
        check_is_fitted(rf_model)
    except NotFittedError:
        raise TlmNotCalibratedError(
            "TLMCalibrated has to be calibrated before the model can be saved, use the .fit() method to calibrate the model."
        )

    # Capture essential state using direct attribute access
    state = {
        "options": model._options,  # noqa: SLF001
        "rf_state": {
            "n_features_in_": (rf_model.n_features_in_ if hasattr(rf_model, "n_features_in_") else None),
            "n_outputs_": (rf_model.n_outputs_ if hasattr(rf_model, "n_outputs_") else None),
            "estimators_": (rf_model.estimators_ if hasattr(rf_model, "estimators_") else None),
            "monotonic_cst_": (rf_model.monotonic_cst_ if hasattr(rf_model, "monotonic_cst_") else None),
        },
        "quality_preset": model._quality_preset,  # noqa: SLF001
        "timeout": model._timeout,  # noqa: SLF001
        "verbose": model._verbose,  # noqa: SLF001
        "num_features": model._num_features,  # noqa: SLF001
    }

    # Get skops and save state
    skops = _get_skops()
    with open(filename, "wb") as f:
        f.write(skops.dumps(state))


def load_tlm_calibrated_state(filename: str) -> TLMCalibrated:
    """Load and reconstruct TLMCalibrated model from file.

    Args:
        filename (str): Path to the saved model state file

    Returns:
        TLMCalibrated: A reconstructed TLMCalibrated model with the saved state

    Raises:
        FileNotFoundError: If the specified file does not exist
        ImportError: If skops package is not installed
    """
    # Get skops for loading
    skops = _get_skops()

    # Load state
    try:
        with open(filename, "rb") as f:
            state = skops.loads(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"No saved model state found at: {filename}")

    # Create new model with saved parameters
    model = TLMCalibrated(
        quality_preset=state.get("quality_preset", "medium"),
        options=state.get("options"),
        timeout=state.get("timeout"),
        verbose=state.get("verbose"),
    )

    # Restore num_features directly
    if state.get("num_features") is not None:
        model._num_features = state["num_features"]  # noqa: SLF001

    # Restore RF model attributes
    rf_model = model._rf_model  # noqa: SLF001
    for attr, value in state["rf_state"].items():
        if value is not None:
            setattr(rf_model, attr, value)

    return model
